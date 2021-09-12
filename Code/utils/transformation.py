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

def h_to_3d(intr, E, height, cam_pos=None):
  '''
  [#] 3D reconstruction by get the position along the ray from a given height
  Input : (All torch.tensor dtype)
    1. intr : intersect plane position in shape(batch, seq_len, 3)
    2. E : extrinsic parameters to compute camera position in shape(batch, seq_len, 4, 4)
    3. height : from height prediction in shape(batch, seq_len, 3)
    4. cam_pos (*opt-required when used canonicalize) : camera position after canonicalization
  Output : (All torch.tensor dtype)
    1. xyz : reconstructed position in shape(batch, seq_len, 3)

  ***Function need to have no detach() to prevent the c-graph discontinue***
  '''
  if cam_pos is None:
    cam_pos = pt.tensor(np.linalg.inv(E.cpu().numpy())[..., 0:3, -1]).to(device)

  R = cam_pos - intr
  norm_R = R / (pt.sqrt(pt.sum(R**2, dim=2, keepdims=True)) + 1e-16)
  magnitude = height / (norm_R[..., [1]] + 1e-16)
  xyz = intr + (norm_R * magnitude)
  return xyz


def cast_ray(uv, I, E, cpos):
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
  w = 1664.0
  h = 1088.0
  
  transf = I @ E
  transf_inv = pt.inverse(transf)

  u = ((uv[..., [0]] / w) * 2) - 1
  v = ((uv[..., [1]] / h) * 2) - 1

  ones = pt.ones(u.shape).to(device)

  ndc = pt.cat((u, v, ones, ones), axis=-1).to(device)
  ndc = pt.unsqueeze(ndc, axis=-1)

  cdt = transf_inv @ ndc

  ray = cdt[..., [0, 1, 2], -1] - cpos
  return ray

def ray_to_plane(cpos, ray):
  '''
  [#] Find the intersection points on the plane given ray-vector and camera position
  Input :
      - cam_pos : camera position in shape=(batch, seq_len, 3)
      - ray : ray vector in shape=(batch, seq_len, 3)
  Output : 
      - intr_pts : ray to plane intersection points from camera through the ball tracking
  '''
  intr_pos = cpos + (ray * (-cpos[..., [1]]/ray[..., [1]]))
  return intr_pos


def compute_azimuth(ray):
  '''
  [#] Compute Azimuth from camera position and it's ray(2d tracking)
  Input : 
      1. Ray : ray direction through 2d-tracking (Obtained from ray-casting function) in shape = (batch, seq_len, 3)
      2. Cam : camera position in shape = (batch, seq_len, 3)
  Output : 
      1. Azimuth angle between ray-direction and x-axis (1, 0) in shape = (batch, seq_len, 1)
  '''
  azimuth = pt.atan2(ray[..., [2]], ray[..., [0]]) * 180.0 / np.pi
  azimuth = (azimuth + 360.0) % 360.0
  return azimuth

def compute_elevation(intr, cpos):
  '''
  [#] Compute Elevation from camera position and it's ray(2d tracking)
  Input : 
      1. Ray : ray direction through 2d-tracking (Obtained from ray-casting function) in shape = (batch, seq_len, 3)
  Output : 
      1. Elevation angle between ray-direction and xz-plane (1, 0) in shape = (batch, seq_len, 1)
  '''
  ray = cpos - intr
  elevation = pt.asin(cpos[..., [1]]/(pt.sqrt(pt.sum(ray**2, axis=-1, keepdims=True)) + 1e-16)) * 180.0 / np.pi
  return elevation


def canonicalize_tmp(pts, R):
  '''
  ***Function has no detach() to prevent the grad_fn***
  Input : Ignore (Y dimension cuz we rotate over Y-axis)
      - pts : 3d points in shape=(batch, seq_len, 3)
      - extrinsic : extrinsic in shape=(batch, seq_len, 4, 4)
      - degree : 
          - if given => decanonicalized the points
          - else => Rotate to reference angle
  Output :  (All are torch.tensor dtype)
      - pts_cl : 3d points that canonicalized
      - cam : camera position after canonicalized (To further find the new azimuth)
      - degree : the degree that we rotate the camera
  '''
  #print("pts: ", pts.shape)

  cpos = Einv[..., 0:3, -1]
  if deg is not None:
    # De-canonicalized the world coordinates
    deR = Ry(cpos)
    # Canonicalized points
    pts = pt.unsqueeze(pts, dim=-1)
    pts_decl = deR @ pts
    pts_decl = pt.squeeze(pts_decl, axis=-1)
    # Canonicalized camera
    cpos = pt.unsqueeze(cpos, dim=-1)
    cpos_decl = deR @ cpos
    cpos_decl = pt.squeeze(cpos_decl, dim=-1)

    return pts_decl, cpos_decl, deg

  else:
    # Find the angle to rotate(canonicalize)
    ref_cam = pt.tensor([0.0, 0.0, 1.0]).reshape(1, 1, 3).to(device)
    deg = pt.atan2(ref_cam[..., 2], ref_cam[..., 0]) - pt.atan2(cpos[..., 2], cpos[..., 0])
    
    l_pi = deg > np.pi
    h_pi = deg <= np.pi
    deg[l_pi] = deg[l_pi] - 2 * np.pi
    deg[h_pi] = deg[h_pi] + 2 * np.pi

    R = Ry(-deg)

    # Canonicalize points
    pts = pt.unsqueeze(pts, dim=-1)
    pts_cl = R @ pts
    pts_cl = pt.squeeze(pts_cl, dim=-1)
    # Canonicalize camera
    cam = pt.tensor(cpos).to(device)
    cam = pt.unsqueeze(cam, dim=-1)
    cam_cl = R @ cam
    cam_cl = pt.squeeze(cam_cl, dim=-1)

    return pts_cl, cam_cl, deg
        

def find_R(Einv):
  '''
  Return a rotation matrix given theta(radian)
  Input : 
    1. theta : radian in shape (batch_size, seq_len)
  Output : 
    1. R : rotation matrix from given radian in shape (batch_size, seq_len, 3, 3)
  '''
  cpos = Einv[..., 0:3, -1].cpu().numpy()
  c = cpos[..., [0, 2]] / (np.linalg.norm(cpos[..., [0, 2]], axis=-1, keepdims=True) + 1e-16)
  sin = c[..., 0]
  cos = c[..., 1]
  zeros = np.zeros(sin.shape)
  ones = np.ones(sin.shape)
  R = np.array([[cos, zeros, -sin], 
              [zeros, ones, zeros],
              [sin, zeros, cos]])
  R = np.transpose(R, (2, 3, 0, 1))
  return pt.tensor(R).float().to(device)

def canonicalize(pts, R, inv=False):
  '''
  Input : Ignore (Y dimension cuz we rotate over Y-axis)
    1. pts : 3d points in shape=(batch, seq_len, 3)
    2. R : Rotation matrix from find_R function in shape(batch, seq_len, 3, 3)
    3. inv : inverse the rotation (For decanonicalization)
  Output :  (All are torch.tensor dtype)
    1. pts : 3d points that (de)canonicalized in shape=(batch, seq_len, 3)
  '''
  if inv:
    inv_mat = pt.tensor([[[[1, 1, -1], 
                        [1, 1, 1],
                        [-1, 1, 1]]]]).to(device)
    R = pt.mul(R, inv_mat)

  pts = pt.unsqueeze(pts, dim=-1)
  pts = R @ pts
  return pt.squeeze(pts, dim=-1)

  
def Ry(theta):
  '''
  Return a rotation matrix given theta(radian)
  Input : 
    1. theta : radian in shape (batch_size, seq_len)
  Output : 
    1. R : rotation matrix from given radian in shape (batch_size, seq_len, 3, 3)
  '''
  print(theta.shape)
  theta = theta.cpu().numpy()
  zeros = np.zeros(theta.shape)
  ones = np.ones(theta.shape)
  R = np.array([[np.cos(theta), zeros, np.sin(theta)], 
                [zeros, ones, zeros],
                [-np.sin(theta), zeros, np.cos(theta)]])
  R = pt.tensor(R)

  print(R.shape)
  R = np.transpose(R, (2, 3, 0, 1))
  print(R.shape)
  exit()
  return R