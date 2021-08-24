from numpy.core.einsumfunc import _compute_size_by_dict
import torch as pt
import json
import numpy as np

args=None

# GPU initialization
if pt.cuda.is_available():
  device = pt.device('cuda')
else:
  device = pt.device('cpu')

def share_args(a):
  global args
  args = a

def project_3d(uv, depth, cam_params_dict, device):
  # print(uv.shape, depth.shape)
  depth = depth.view(-1)
  screen_width = cam_params_dict['main']['width']
  screen_height = cam_params_dict['main']['height']
  I_inv = cam_params_dict['main']['I_inv']
  E_inv = cam_params_dict['main']['E_inv']
  uv = pt.div(uv, pt.tensor([screen_width, screen_height]).to(device)) # Normalize : (width, height) -> (-1, 1)
  uv = (uv * 2.0) - pt.ones(size=(uv.size()), dtype=pt.float32).to(device) # Normalize : (width, height) -> (-1, 1)
  uv = (uv.t() * depth).t()   # Normalize : (-1, 1) -> (-depth, depth) : Camera space (x', y', d, 1)
  uv = pt.stack((uv[:, 0], uv[:, 1], depth, pt.ones(depth.shape[0], dtype=pt.float32).to(device)), axis=1) # Stack the screen with depth and w ===> (x, y, depth, 1)
  uv = ((E_inv @ I_inv) @ uv.t()).t() # Reprojected

  return uv[:, :3]

def centering(xyz, cam_params_dict, device, rev=False):
  x = cam_params_dict['center_offset']['x']
  y = cam_params_dict['center_offset']['y']
  z = cam_params_dict['center_offset']['z']

  center_offset = pt.tensor([[[x, y, z]]]).to(device)
  if not rev:
    xyz = xyz - center_offset
  else:
    xyz = xyz + center_offset

  return xyz

def get_cam_params_dict(cam_params_file, device):
  '''
  Return the cameras parameters use in reconstruction
  '''
  cam_params_dict = {}
  cam_use = ['main'] + args.multiview_loss
  with open(cam_params_file) as cam_params_json:
    cam_params_file = json.load(cam_params_json)
    for each_cam_use in cam_use:
      cam_unity_key = '{}PitchCameraParams'.format(each_cam_use)
      cam_params_dict[each_cam_use] = {}
      # Extract each camera parameters
      cam_params = dict({'projectionMatrix':cam_params_file[cam_unity_key]['projectionMatrix'], 'worldToCameraMatrix':cam_params_file[cam_unity_key]['worldToCameraMatrix'], 'width':cam_params_file[cam_unity_key]['width'], 'height':cam_params_file[cam_unity_key]['height']})
      projection_matrix = np.array(cam_params['projectionMatrix']).reshape(4, 4)
      projection_matrix = pt.tensor([projection_matrix[0, :], projection_matrix[1, :], projection_matrix[3, :], [0, 0, 0, 1]], dtype=pt.float32)
      cam_params_dict[each_cam_use]['I'] = projection_matrix.to(device)
      cam_params_dict[each_cam_use]['I_inv'] = pt.inverse(projection_matrix).to(device)

      cam_params_dict[each_cam_use]['E'] = pt.tensor(cam_params['worldToCameraMatrix']).view(4, 4).to(device)
      cam_params_dict[each_cam_use]['E_inv'] = pt.inverse(pt.tensor(cam_params['worldToCameraMatrix']).view(4, 4)).to(device)
      cam_params_dict[each_cam_use]['width'] = cam_params['width']
      cam_params_dict[each_cam_use]['height'] = cam_params['height']

    if 'center_offset' in cam_params_file.keys():
      cam_params_dict['center_offset'] = cam_params_file['center_offset']
    else:
      cam_params_dict['center_offset'] = dict({'x':0.0, 'y':0.0, 'z':0.0})

  return cam_params_dict

def project_2d(world, cam_params_dict, normalize=False):
  world = pt.cat((world, pt.ones(world.shape[0], world.shape[1], 1).to(device)), dim=-1)
  I = cam_params_dict['I']
  E = cam_params_dict['E']
  width = cam_params_dict['width']
  height = cam_params_dict['height']
  transformation = (I @ E)
  ndc = (world @ transformation.t())
  if normalize:
    u = (ndc[..., [0]]/ndc[..., [2]] + 1) * .5
    v = (ndc[..., [1]]/ndc[..., [2]] + 1) * .5
  else:
    u = (((ndc[..., [0]]/ndc[..., [2]] + 1) * .5) * width)
    v = (((ndc[..., [1]]/ndc[..., [2]] + 1) * .5) * height)
  d = ndc[..., [2]]
  return u, v, d

def IE_array(trajectory, col):
  '''
  Input :
      1. trajectory : in shape (seq_len, features)
      2. col : Index of intrinsic/extrinsic column
  Output :
      1. 4x4 Matrix : in shape (seq_len, 4, 4)
  '''
  mat = [t[col].reshape(-1, 4, 4) for t in trajectory]
  mat = np.array(mat)
  mat = np.squeeze(mat, axis=1)
  return mat

def h_to_3d(intr, E, height):
  cam_pos = np.linalg.inv(E.cpu().numpy())[..., 0:3, -1]
  R = pt.tensor(cam_pos - intr).to(device)
  norm_R = R / (pt.sqrt(pt.sum(R**2, dim=2, keepdims=True)) + 1e-16)
  magnitude = height / (norm_R[..., [1]] + 1e-16)
  xyz = pt.tensor(intr).to(device) + (norm_R * magnitude)
  return xyz
    

def cast_ray(uv, I, E):
  '''
  Casting a ray given UV-coordinates, intrinisc, extrinsic
  Input : 
      - UV : 2d screen points in shape=(batch, seq_len, 2)
      - I : camera position in shape=(batch, seq_len, 4, 4)
      - E : camera position in shape=(batch, seq_len, 4, 4)
  Output :
      - ray : Ray direction (batch, seq_len, 3)
      (Can check with ray.x, ray.y, ray.z for debugging)
  '''
  #print("UV: : ", uv.shape)
  #print("I : ", I.shape)
  #print("E : ", E.shape)

  w = 1664.0
  h = 1088.0
  
  transf = I @ E
  transf_inv = np.linalg.inv(transf.cpu().numpy())

  u = ((uv[..., [0]] / w) * 2) - 1
  v = ((uv[..., [1]] / h) * 2) - 1

  ones = np.ones(u.shape)

  ndc = np.concatenate((u, v, ones, ones), axis=-1)
  ndc = np.expand_dims(ndc, axis=-1)

  ndc = transf_inv @ ndc

  cam_pos = np.linalg.inv(E.cpu().numpy())[..., 0:3, -1]
  ray = ndc[..., [0, 1, 2], -1] - cam_pos
  return ray

def ray_to_plane(E, ray):
  '''
  Find the intersection points on the plane given ray-vector and camera position
  Input :
      - cam_pos : camera position in shape=(batch, seq_len, 3)
      - ray : ray vector in shape=(batch, seq_len, 3)
  Output : 
      - intr_pts : ray to plane intersection points from camera through the ball tracking
  '''
  c_pos = np.linalg.inv(E.cpu().numpy())[..., 0:3, -1]
  normal = np.array([0, 1, 0])
  p0 = np.array([0, 0, 0])
  ray_norm = ray / np.linalg.norm(ray, axis=-1, keepdims=True)

  t = np.dot((p0-c_pos), normal) / np.dot(ray_norm, normal)
  t = np.expand_dims(t, axis=-1)

  intr_pos = c_pos + (ray_norm * t)

  return intr_pos


def compute_azimuth(ray):
  '''
  Compute Azimuth from camera position and it's ray(2d tracking)
  Input : 
      1. Ray : ray direction through 2d-tracking (Obtained from ray-casting function) in shape = (batch, seq_len, 3)
      2. Cam : camera position in shape = (batch, seq_len, 3)
  Output : 
      1. Azimuth angle between ray-direction and x-axis (1, 0) in shape = (batch, seq_len, 1)
  '''
  azimuth = np.arctan2(ray[..., [2]], ray[..., [0]]) * 180.0 / np.pi
  azimuth = (azimuth + 360.0) % 360.0
  return azimuth

def compute_elevation(intr, E):
  '''
  Compute Elevation from camera position and it's ray(2d tracking)
  Input : 
      1. Ray : ray direction through 2d-tracking (Obtained from ray-casting function) in shape = (batch, seq_len, 3)
  Output : 
      1. Elevation angle between ray-direction and xz-plane (1, 0) in shape = (batch, seq_len, 1)
  '''
  c_pos = np.linalg.inv(E.cpu().numpy())[..., 0:3, -1]
  ray = c_pos - intr
  elevation = np.arcsin(c_pos[..., [1]]/(np.sqrt(np.sum(ray**2, axis=-1, keepdims=True)) + 1e-16)) * 180.0 / np.pi
  return elevation


def canonicalize(pts, E, deg=None):
  '''
  Input : Ignore (Y dimension cuz we rotate over Y-axis)
      - pts : 3d points in shape=(batch, seq_len, 3)
      - extrinsic : extrinsic in shape=(batch, seq_len, 4, 4)
      - degree : 
          - if given => decanonicalized the points
          - else => Rotate to reference angle
  Output :
      - pts_cl : 3d points that canonicalized
      - cam : camera position after canonicalized (To further find the new azimuth)
      - degree : the degree that we rotate the camera
  '''
  #print("pts: ", pts.shape)

  cam = np.linalg.inv(E.cpu().numpy())[..., 0:3, -1]
  pts = pt.tensor(pts).float().to(device)
  if deg is not None:
    # De-canonicalized the world coordinates
    deR = Ry(deg)
    # Canonicalized points
    pts = pt.unsqueeze(pts, dim=-1)
    pts_decl = deR @ pts
    pts_decl = pt.squeeze(pts_decl, axis=-1)
    # Canonicalized camera
    cam = pt.unsqueeze(pt.tensor(cam).to(device), dim=-1)
    cam_decl = deR @ cam
    cam_decl = pt.squeeze(cam_decl, dim=-1)

    return pts_decl, cam_decl, deg


  else:
    # Find the angle to rotate(canonicalize)
    ref_cam = np.array([0.0, 0.0, 1.0]).reshape(1, 1, 3)
    angle = np.arctan2(ref_cam[..., 2], ref_cam[..., 0]) - np.arctan2(cam[..., 2], cam[..., 0])
    
    l_pi = angle > np.pi
    h_pi = angle <= np.pi
    angle[l_pi] = angle[l_pi] - 2 * np.pi
    angle[h_pi] = angle[h_pi] + 2 * np.pi

    R = Ry(-angle)

    # Canonicalize points
    pts = pt.unsqueeze(pts, dim=-1)
    pts_cl = R @ pts
    pts_cl = pt.squeeze(pts_cl, dim=-1)
    # Canonicalize camera
    cam = pt.unsqueeze(pt.tensor(cam).to(device), dim=-1)
    cam_cl = R @ cam
    cam_cl = pt.squeeze(cam_cl, dim=-1)

    return pts_cl, cam_cl, angle
        
def Ry(theta):
  '''
  Return a rotation matrix given theta
  '''
  zeros = np.zeros(theta.shape)
  ones = np.ones(theta.shape)

  R = np.array([[np.cos(theta), zeros, np.sin(theta)], 
                [zeros, ones, zeros],
                [-np.sin(theta), zeros, np.cos(theta)]])

  R = np.transpose(R, (2, 3, 0, 1))
  return pt.tensor(R).float().to(device)