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
  transf = I @ E
  transf_inv = pt.inverse(transf)

  u = ((uv[..., [0]] / args.w) * 2) - 1
  v = ((uv[..., [1]] / args.h) * 2) - 1

  ones = pt.ones(u.shape).to(device)

  ndc = pt.cat((u, v, ones, ones), axis=-1).to(device)
  ndc = pt.unsqueeze(ndc, axis=-1)

  cdt = transf_inv @ ndc

  ray = cdt[..., [0, 1, 2], -1] - cpos
  return ray

def ray_to_plane(cpos, ray, plane):
  '''
  [#] Find the intersection points on the plane given ray-vector and camera position
  Input :
      - cam_pos : camera position in shape=(batch, seq_len, 3)
      - ray : ray vector in shape=(batch, seq_len, 3)
  Output : 
      - intr_pts : ray to plane intersection points from camera through the ball tracking
  '''
  if plane == 'horizontal':
    intr_pos = cpos + (ray * (-cpos[..., [1]]/(ray[..., [1]] + 1e-16)))
  elif plane == 'vertical':
    intr_pos = cpos + (ray * (-cpos[..., [2]]/(ray[..., [2]] + 1e-16)))
  else:
    raise ValueError("A Wrong plane was given.")

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

def find_R(Einv):
  '''
  Return a rotation matrix given theta(radian)
  Input : 
    1. theta : radian in shape (batch_size, seq_len)
  Output : 
    1. R : rotation matrix rotate to [x=1, z=0] in shape (batch_size, seq_len, 3, 3)
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
  Canonicalize any points given R matrix
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

def projection_2d(pts, cam_dict, normalize=False):
  '''
  Projection from 3d to 2d
  Input :
    1. pts : 3d points in shape=(batch, seq_len, 3)
    2. cam_dict : contains I, E, E_inv and cpos => {I, E, E_inv, tracking, cpos}
    3. normalize : True = ndc, False = screen
  Output : 
    1. u : u-coordinates in shape=(batch, seq_len, 1)
    2. v : v-coordinates in shape=(batch, seq_len, 1)
    3. d : depth in shape=(batch, seq_len, 1)
  '''

  ones = pt.ones(size=(pts.shape[0], pts.shape[1], 1)).to(device)
  pts = pt.unsqueeze(pt.cat((pts, ones), dim=-1), dim=-1)
  I = cam_dict['I']
  E = cam_dict['E']
  scr = I @ E @ pts
  scr = pt.squeeze(scr, dim=-1)
  if normalize:
    u = ((scr[..., [0]]/scr[..., [2]] + 1e-16) + 1) * .5
    v = ((scr[..., [1]]/scr[..., [2]] + 1e-16) + 1) * .5
  else:
    u = (((scr[..., [0]]/(scr[..., [2]] + 1e-16) + 1) * .5) * args.w)
    v = (((scr[..., [1]]/(scr[..., [2]] + 1e-16) + 1) * .5) * args.h)

  d = scr[..., [2]]
  return u, v, d

def reconstruct(height, cam_dict, recon_dict, canon_dict):
  '''
  Reconstruct the 3d points from predicted height
  Input : 
    1. height : predicted height in shape(batch_size, seq_len, 1)
    2. cam_dict : contains ['I', 'E', 'Einv', 'tracking']
    3. recon_dict : contains 
      - 'clean' : for clean reconstruction
      - 'noisy' : for noisy reconstruction
    4. canon_dict : contains ['cam_cl', 'R'] to be used in reconstruction and canonicalize
  Output : 
    1. xyz : reconstructed xyz in shape(batch_size, seq_len, 3)
  '''

  if args.canonicalize:
    intr_clean = canonicalize(pts=recon_dict['clean'], R=canon_dict['R'])
    intr_noisy = canonicalize(pts=recon_dict['noisy'], R=canon_dict['R'])
  else:
    intr_clean = recon_dict['clean']
    intr_noisy = recon_dict['noisy']

  if args.recon == 'clean':
    xyz = h_to_3d(height=height, intr=intr_clean, E=cam_dict['E'], cam_pos=canon_dict['cam_cl'])
  elif args.recon == 'noisy':
    xyz = h_to_3d(height=height, intr=intr_noisy, E=cam_dict['E'], cam_pos=canon_dict['cam_cl'])
  
  return xyz