from numpy.lib import utils
from Code.utils.utils_func import generate_input
import torch as pt
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math
from tqdm import tqdm
sys.path.append(os.path.realpath('../..'))
import utils.transformation as utils_transform
import utils.utils_func as utils_func
import utils.loss as utils_loss
args = None

# GPU initialization
if pt.cuda.is_available():
  device = pt.device('cuda')
else:
  device = pt.device('cpu')

def share_args(a):
  global args
  args = a


def train_mode(model_dict):
  for model in model_dict.keys():
    model_dict[model].train()

def eval_mode(model_dict):
  for model in model_dict.keys():
    model_dict[model].eval()

def fw_pass(model_dict, input_dict, cam_dict, gt_dict):
  '''
  Forward Pass to the whole pipeline
  Input :
      1. model_dict : contains modules in the pipeline
      2. input_dict : input to the model => {input, lengths, mask}
      3. cam_dict : contains I, E, E_inv and cpos => {I, E, E_inv, tracking, cpos}
      4. gt_dict : contains ground truth trajectories => {gt, lengths, mask}
  Output : 
      1. pred_dict : prediction of each moduel and reconstructed xyz
      2. in_f : in case of we used noisy (u, v) 
  '''

  # Input
  in_f = input_dict['input']

  # Add noise
  if args.noise:
    in_f, ray = utils_func.add_noise(cam_dict=cam_dict, in_f=in_f)

  # Generate Input directly from tracking
  in_f, ray = generate_input(in_f=in_f, cam_dict=cam_dict)


  if args.canonicalize:
    in_f_cl, _, angle = utils_transform.canonicalize(pts=in_f[..., [0, 1, 2]], E=cam_dict['E'])
    ray_cl, _, _ = utils_transform.canonicalize(pts=ray[..., [0, 1, 2]], E=cam_dict['E'])
    azim_cl = utils_transform.compute_azimuth(ray=ray_cl.cpu().numpy())
    in_f = np.concatenate((in_f_cl.cpu().numpy(), azim_cl, in_f[..., [3]]), axis=2)
    #in_f_decl, _, _ = utils_transform.canonicalize(pts=in_f_cl[..., [0, 1, 2]], E=cam_dict['E'], deg=angle)
    #ray_decl, _, _ = utils_transform.canonicalize(pts=ray_cl[..., [0, 1, 2]], E=cam_dict['E'], deg=angle)
  

  intr_recon = in_f[..., [0, 1, 2]].copy()
  in_f = input_manipulate(in_f=in_f)
  
  if 'height' in args.pipeline:
    height, _ = model_dict['height'](in_f=in_f, lengths=input_dict['lengths']-1 if args.i_s == 'dt' else input_dict['lengths'])
    
  #height = output_space(height)
  height = gt_dict['gt'][..., [1]]
  pred_xyz = utils_transform.h_to_3d(height=height, intr=intr_recon, E=cam_dict['E'])

  # Decoanonicalize
  if args.canonicalize:
    pred_xyz, _, _ = utils_transform.canonicalize(pts=pred_xyz, E=cam_dict['E'], deg=angle)

  print(pred_xyz - gt_dict['gt'])
  print(pt.max(pred_xyz - gt_dict['gt']))
  print(pt.mean(pred_xyz - gt_dict['gt']))
  exit()
  
  return pred_xyz
      

def input_manipulate(in_f):
  '''
  Prepare input to have correct space(t, dt, tdt), sin-cos
  Input : 
    1. in_f : input features in shape (batch, seq_len, 5)
  Output : 
    1. in_f : input features after change into specified space(t, dt, tdt) and sin-cos
  '''

  if args.sc == 'azim':
    azim_sc = np.concatenate(np.sin(in_f[..., 4]), np.cos(in_f[..., 4]), axis=2)
    in_f = np.concatenate((in_f[..., [0, 1, 2]], in_f[..., [3], azim_sc]), axis=2)
  elif args.sc == 'elev':
    elev_sc = np.concatenate(np.sin(in_f[..., 3]), np.cos(in_f[..., 3]), axis=2)
    in_f = np.concatenate((in_f[..., [0, 1, 2]], elev_sc, in_f[..., [4]]), axis=2)
  elif args.sc == 'both':
    azim_sc = np.concatenate(np.sin(in_f[..., 4]), np.cos(in_f[..., 4]), axis=2)
    elev_sc = np.concatenate(np.sin(in_f[..., 3]), np.cos(in_f[..., 3]), axis=2)
    in_f = np.concatenate((in_f[..., [0, 1, 2]], elev_sc, azim_sc), axis=2)

  in_f = input_space(in_f)
  in_f = pt.tensor(in_f).float().to(device)

  return in_f
    
def input_space(in_f):
  '''
  Input : 
    1. in_f : input features(t-space) in shape (batch, seq_len, 5) -> (x, y, z, elev, azim)
  Output :
    1. in_f : input features in t/dt/t_dt-space in shape(batch, seq_len, _)
  '''
  if args.i_s == 'dt':
    in_f = in_f[:, 1:, :] - in_f[:, :-1, :]
  elif args.i_s == 't':
    in_f = in_f
  elif args.i_s == 't_dt':
    dt = in_f[:, 1:, :] - in_f[:, :-1, :]
    t0_pad = np.zeros(shape=(in_f.shape[0], 1, in_f.shape[2]))
    dt = np.concatenate((t0_pad, dt), axis=1)
    in_f = np.concatenate((in_f, dt), axis=2)

  return in_f


def output_space(height):
  '''
  Prepare input to have correct space(t, dt, tdt), sin-cos
  Input : 
    1. in_f : input features in shape (batch, seq_len, 5)
  Output : 
    1. in_f : input features after change into specified space(t, dt, tdt) and sin-cos
  '''
  pass


def calculate_loss():
  # Calculate loss term
  ######################################
  ############# Trajectory #############
  ######################################
  if 'depth' in args.pipeline or 'refinement' in args.pipeline:
    trajectory_loss = utils_loss.TrajectoryLoss(pred=pred_xyz[..., [0, 1, 2]], gt=gt_dict['xyz'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])
    gravity_loss = utils_loss.GravityLoss(pred=pred_xyz[..., [0, 1, 2]], gt=gt_dict['xyz'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])
    below_ground_loss = utils_loss.BelowGroundPenalize(pred=pred_xyz[..., [0, 1, 2]], gt=gt_dict['xyz'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])

  else:
    trajectory_loss = pt.tensor(0.).to(device)
    gravity_loss = pt.tensor(0.).to(device)
    below_ground_loss = pt.tensor(0.).to(device)

  return loss_dict, loss