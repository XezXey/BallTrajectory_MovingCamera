from Code.utils.utils_func import generate_input
import torch as pt
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
sys.path.append(os.path.realpath('../..'))
# Utils
import utils.transformation as utils_transform
import utils.utils_func as utils_func
import utils.loss as utils_loss
# Models
from models.optimization import Optimization
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

'''
def fw_pass(model_dict, input_dict, cam_dict, gt_dict):
  Forward Pass to the whole pipeline
  Input :
      1. model_dict : contains modules in the pipeline
      2. input_dict : input to the model => {input, lengths, mask}
      3. cam_dict : contains I, E, E_inv and cpos => {I, E, E_inv, tracking, cpos}
      4. gt_dict : contains ground truth trajectories => {gt, lengths, mask}
  Output : 
      1. pred_dict : prediction of each moduel and reconstructed xyz
      2. in_f : in case of we used noisy (u, v) 

  # Input
  in_f = input_dict['input']

  # Prediction dict
  pred_dict = {}

  # Add noise
  if args.noise:
    in_f, ray = utils_func.add_noise(cam_dict=cam_dict, in_f=in_f)

  # Generate Input directly from tracking
  in_f, ray = generate_input(in_f=in_f, cam_dict=cam_dict)

  # Canonicalize
  if args.canonicalize:
    in_f_cl, cam_cl, angle = utils_transform.canonicalize(pts=in_f[..., [0, 1, 2]], E=cam_dict['E'])
    ray_cl, _, _ = utils_transform.canonicalize(pts=ray[..., [0, 1, 2]], E=cam_dict['E'])
    azim_cl = utils_transform.compute_azimuth(ray=ray_cl.cpu().numpy())
    azim_cl = pt.tensor(azim_cl).float().to(device)
    in_f = pt.cat((in_f_cl, azim_cl, in_f[..., [3]]), dim=2)

  intr_recon = in_f[..., [0, 1, 2]]
  in_f = input_manipulate(in_f=in_f)

  search_h = None
  # Aug
  if args.augment:
    if args.optim_h:
      search_h = {}
      optim_first_h = Optimization(shape=(in_f.shape[0], 1, 1), n_optim=50)
      optim_last_h = Optimization(shape=(in_f.shape[0], 1, 1), n_optim=50)
      train_mode(model_dict=model_dict)
      optim_first_h.train()
      optim_last_h.train()
      search_h['first_h'] = optim_first_h.get_params()
      search_h['last_h'] = optim_last_h.get_params()
    else:
      search_h = {}
      search_h['first_h'] = gt_dict['gt'][:, [0], [1]]
      search_h['last_h'] = pt.stack([gt_dict['gt'][i, [input_dict['lengths'][i]-1], [1]] for i in range(gt_dict['gt'].shape[0])])
      search_h['first_h'] = pt.unsqueeze(search_h['first_h'], dim=-1)
      search_h['last_h'] = pt.unsqueeze(search_h['last_h'], dim=-1)
  
  for i in range(10):
    if 'height' in args.pipeline:
      pred_h, _ = model_dict['height'](in_f=in_f, lengths=input_dict['lengths']-1 if args.i_s == 'dt' else input_dict['lengths'])
      pred_dict['h'] = pred_h

    height = output_space(pred_h, lengths=input_dict['lengths'], search_h=search_h)

    pred_xyz = utils_transform.h_to_3d(height=height, intr=intr_recon, E=cam_dict['E'], cam_pos=cam_cl if args.canonicalize else None)

    # Decoanonicalize
    if args.canonicalize:
      pred_xyz, _, _ = utils_transform.canonicalize(pts=pred_xyz, E=cam_dict['E'], deg=angle)

    pred_dict['xyz'] = pred_xyz

    _, loss = optimization_loss(input_dict=input_dict, gt_dict=gt_dict, pred_dict=pred_dict)
    optim_first_h(loss)
    optim_last_h(loss)

  return pred_dict, in_f
'''

def uv_to_inf(cam_dict):
  '''
  Create the input features from uv-tracking
  Input : 
    1. cam_dict : contains [I, E, Einv, tracking]
  Output :
    1. recon_dict : contains 
      - 'clean' : for clean reconstruction
      - 'noisy' : for noisy reconstruction
  '''
  # Add noise & generate input
  in_f_noisy, ray_noisy = utils_func.add_noise(cam_dict=cam_dict)
  # Generate input
  in_f_raw, ray_raw = utils_func.generate_input(cam_dict=cam_dict)

  if args.noise:
    in_f, ray = in_f_noisy, ray_noisy
  else:
    in_f, ray = in_f_raw, ray_raw

  recon_dict = {'clean' : in_f_raw[..., [0, 1, 2]], 'noisy': in_f_noisy[..., [0, 1, 2]]}
  return in_f, ray, recon_dict

def canonicalize_features(Einv, intr, ray, in_f):
  '''
  Canonicalize all features
  Input : 
    1. Einv : To compute a rotation matrix in shape (batch_size, seq_len, 4, 4)
    2. intr : intersection points in shape(batch_size, seq_len, 3)
    3. ray : ray direction in shape(batch_size, seq_len, 3)
    4. in_f : input features from uv in shape(batch_size, seq_len, 5) - (intr, elev, azim)
  Output :
    1. canon_dict : contains ['cam_cl', 'R'] to be used in reconstruction and canonicalize
    2. in_f : the new input features after canonicalized
  '''
  R = utils_transform.find_R(Einv=Einv)
  intr_cl = utils_transform.canonicalize(pts=intr, R=R)
  cam_cl = utils_transform.canonicalize(pts=Einv[..., 0:3, -1], R=R)
  ray_cl = utils_transform.canonicalize(pts=ray[..., [0, 1, 2]], R=R)
  azim_cl = utils_transform.compute_azimuth(ray=ray_cl)
  in_f = pt.cat((intr_cl, azim_cl, in_f[..., [3]]), dim=2)
  canon_dict = {'cam_cl' : cam_cl, 'R' : R}

  return canon_dict, in_f

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

  # Prediction dict
  pred_dict = {}

  in_f, ray, recon_dict = uv_to_inf(cam_dict)

  # Canonicalize
  if args.canonicalize:
    canon_dict, in_f = canonicalize_features(Einv=cam_dict['Einv'], intr=in_f[..., [0, 1, 2]], ray=ray, in_f=in_f)
  else:
    canon_dict = {'cam_cl' : None, 'R' : None}

  in_f = input_manipulate(in_f=in_f)

  search_h = None
  # Augmentation
  if args.augment:
    search_h = {}
    search_h['first_h'] = gt_dict['gt'][:, [0], [1]]
    search_h['last_h'] = pt.stack([gt_dict['gt'][i, [input_dict['lengths'][i]-1], [1]] for i in range(gt_dict['gt'].shape[0])])
    search_h['first_h'] = pt.unsqueeze(search_h['first_h'], dim=-1)
    search_h['last_h'] = pt.unsqueeze(search_h['last_h'], dim=-1)
  # Optimizae for initial height
  if args.optim_h:
    search_h = {}
    optim_first_h = Optimization(shape=(in_f.shape[0], 1, 1), n_optim=50)
    optim_last_h = Optimization(shape=(in_f.shape[0], 1, 1), n_optim=50)
    train_mode(model_dict=model_dict)
    optim_first_h.train()
    optim_last_h.train()
    search_h['first_h'] = optim_first_h.get_params()
    search_h['last_h'] = optim_last_h.get_params()
  

  if 'flag' in args.pipeline:
    pred_flag, _ = model_dict['flag'](in_f=in_f, lengths=input_dict['lengths']-1 if args.i_s == 'dt' else input_dict['lengths'])
    pred_dict['flag'] = pred_flag
    in_f = pt.cat((in_f, pred_flag), dim=-1)

  if 'height' in args.pipeline:
    pred_h, _ = model_dict['height'](in_f=in_f, lengths=input_dict['lengths']-1 if args.i_s == 'dt' else input_dict['lengths'])
    pred_dict['h'] = pred_h

  height = output_space(pred_h, lengths=input_dict['lengths'], search_h=search_h)

  xyz = reconstruct(height, cam_dict, recon_dict, canon_dict)

  if 'refinement' in args.pipeline:
    pred_refoff, _ = model_dict['refinement'](in_f=xyz, lengths=input_dict['lengths'])
    pred_dict['refine_offset'] = pred_refoff
    xyz_refined = xyz + pred_refoff
  else:
    xyz_refined = None
    
  # Decoanonicalize
  if args.canonicalize:
    xyz = utils_transform.canonicalize(pts=xyz, R=canon_dict['R'], inv=True)
    if xyz_refined is not None:
      xyz_refined = utils_transform.canonicalize(pts=xyz_refined, R=canon_dict['R'], inv=True)

  pred_dict['xyz'] = xyz
  pred_dict['xyz_refined'] = xyz_refined

  return pred_dict, in_f

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
    intr_clean = utils_transform.canonicalize(pts=recon_dict['clean'], R=canon_dict['R'])
    intr_noisy = utils_transform.canonicalize(pts=recon_dict['noisy'], R=canon_dict['R'])
  else:
    intr_clean = recon_dict['clean']
    intr_noisy = recon_dict['noisy']

  if args.recon == 'clean':
    xyz = utils_transform.h_to_3d(height=height, intr=intr_clean, E=cam_dict['E'], cam_pos=canon_dict['cam_cl'])
  elif args.recon == 'noisy':
    xyz = utils_transform.h_to_3d(height=height, intr=intr_noisy, E=cam_dict['E'], cam_pos=canon_dict['cam_cl'])
  
  return xyz
      
def input_manipulate(in_f):
  '''
  Prepare input to have correct space(t, dt, tdt), sin-cos
  Input : 
    1. in_f : input features in shape (batch, seq_len, 5)
  Output : 
    1. in_f : input features after change into specified space(t, dt, tdt) and sin-cos
  '''

  if args.sc == 'azim':
    azim_sc = pt.cat((pt.sin(in_f[..., 4]), pt.cos(in_f[..., 4])), axis=2)
    in_f = pt.cat((in_f[..., [0, 1, 2]], in_f[..., [3], azim_sc]), axis=2)
  elif args.sc == 'elev':
    elev_sc = pt.cat((pt.sin(in_f[..., 3]), pt.cos(in_f[..., 3])), axis=2)
    in_f = pt.cat((in_f[..., [0, 1, 2]], elev_sc, in_f[..., [4]]), axis=2)
  elif args.sc == 'both':
    azim_sc = pt.cat((pt.sin(in_f[..., 4]), pt.cos(in_f[..., 4])), axis=2)
    elev_sc = pt.cat((pt.sin(in_f[..., 3]), pt.cos(in_f[..., 3])), axis=2)
    in_f = pt.cat((in_f[..., [0, 1, 2]], elev_sc, azim_sc), axis=2)

  in_f = input_space(in_f)

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

def output_space(pred_h, lengths, search_h=None):
  '''
  Aggregate the height-prediction into (t, dt)
  Input : 
    1. height : height in shape (batch, seq_len, 1)
    2. lengths : lengths of each seq to determine the wegiht size, and position to reverse the seq(always comes in t-space)
    3. search_h(optional) : In scenario which is trajectory didn't start/end on the ground. (Handling by search for h)
      in shape {'first_h' : (batch, 1, 1), 'last_h' : (batch, 1, 1)}
  Output : 
    1. height :height after aggregation into t-space
  '''
  if args.o_s == 't':
    return pred_h 
    
  elif args.o_s == 'dt':
    if args.i_s == 't_dt':
      raise NotImplemented
      
    elif args.i_s == 'dt':
      pred_h = pred_h

    w_ramp = utils_func.construct_w_ramp(weight_template=pt.zeros(size=(pred_h.shape[0], pred_h.shape[1]+1, 1)), lengths=lengths)

    if search_h is None:
      first_h = pt.zeros(size=(pred_h.shape[0], 1, 1)).to(device)
      last_h = pt.zeros(size=(pred_h.shape[0], 1, 1)).to(device)
    else:
      first_h = search_h['first_h']
      last_h = search_h['last_h']

    # forward aggregate
    h_fw = utils_func.cumsum(seq=pred_h, t_0=first_h)
    # backward aggregate
    pred_h_bw = utils_func.reverse_masked_seq(seq=-pred_h, lengths=lengths-1) # This fn required len(seq) of dt-space
    #print(pt.cat((pred_h, pred_h_bw), dim=-1))
    h_bw = utils_func.cumsum(seq=pred_h_bw, t_0=last_h)
    h_bw = utils_func.reverse_masked_seq(seq=h_bw, lengths=lengths) # This fn required len(seq) of t-space(after cumsum)
    #print(pt.cat((h_fw[0], h_bw[0], w_ramp[0]), dim=1))
    #print(pt.cat((h_fw[1], h_bw[1], w_ramp[1]), dim=1))
    height = pt.sum(pt.cat((h_fw, h_bw), dim=2) * w_ramp, dim=2, keepdims=True)
      
    return height

def training_loss(input_dict, gt_dict, pred_dict, cam_dict, anneal_w):
  '''
  Calculate loss
  '''
  ######################################
  ############# Trajectory #############
  ######################################
  if (args.annealing) and ('refinement' in args.pipeline):
    # Annealing
    trajectory_loss = utils_loss.TrajectoryLoss(pred=pred_dict['xyz'], gt=gt_dict['gt'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])
    trajectory_loss_refined = utils_loss.TrajectoryLoss(pred=pred_dict['xyz_refined'], gt=gt_dict['gt'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])
    trajectory_loss = trajectory_loss + trajectory_loss_refined * anneal_w
  elif ('refinement' in args.pipeline):
    # No annealing
    trajectory_loss = utils_loss.TrajectoryLoss(pred=pred_dict['xyz_refined'], gt=gt_dict['gt'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])
  else:
    # No refinement
    trajectory_loss = utils_loss.TrajectoryLoss(pred=pred_dict['xyz'], gt=gt_dict['gt'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])

  ######################################
  ############# Below GND ##############
  ######################################
  below_ground_loss = utils_loss.BelowGroundLoss(pred=pred_dict['xyz'], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])
  #below_ground_loss = pt.tensor(0.).to(device)

  ######################################
  ############## Gravity ###############
  ######################################
  #gravity_loss = utils_loss.GravityLoss(pred=pred_dict['xyz'], gt=gt_dict['gt'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])
  gravity_loss = pt.tensor(0.).to(device)

  ######################################
  ############### Flag #################
  ######################################
  if 'flag' not in pred_dict.keys() or args.env != 'unity':
    flag_loss = pt.tensor(0.).to(device)
  else:
    zeros = pt.zeros((pred_dict['flag'].shape[0], 1, 1)).to(device)
    flag_loss = utils_loss.EndOfTrajectoryLoss(pred=pt.cat((zeros, pred_dict['flag']), dim=1), gt=gt_dict['gt'][..., [3]], mask=gt_dict['mask'][..., [3]], lengths=gt_dict['lengths'])

  ######################################
  ########### Reprojection #############
  ######################################
  if ('refinement' in args.pipeline):
    reprojection_loss = utils_loss.ReprojectionLoss(pred=pred_dict['xyz_refined'], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'], cam_dict=cam_dict)
  else:
    #reproj_loss = pt.tensor(0.).to(device)
    reprojection_loss = utils_loss.ReprojectionLoss(pred=pred_dict['xyz'], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'], cam_dict=cam_dict)

  # Combined all losses term
  loss = trajectory_loss + gravity_loss + below_ground_loss + flag_loss + reprojection_loss
  loss_dict = {"Trajectory Loss":trajectory_loss.item(),
               "Gravity Loss":gravity_loss.item(),
               "BelowGnd Loss":below_ground_loss.item(),
               "Flag Loss":flag_loss.item(),
               "Reprojection Loss":reprojection_loss.item()
               }

  return loss_dict, loss

def optimization_loss(input_dict, gt_dict, pred_dict):
  # Optimization loss term
  ######################################
  ############# Trajectory #############
  ######################################
  trajectory_loss = utils_loss.TrajectoryLoss(pred=pred_dict['xyz'], gt=gt_dict['gt'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])
  below_ground_loss = utils_loss.BelowGroundPenalize(pred=pred_dict['xyz'], mask=input_dict['mask'], lengths=input_dict['lengths'])

  #gravity_loss = pt.tensor(0.).to(device)
  #below_ground_loss = pt.tensor(0.).to(device)

  loss = below_ground_loss + trajectory_loss
  loss_dict = {"Trajectory Loss":trajectory_loss.item(),
              "BelowGnd Loss":below_ground_loss.item()}

  return loss_dict, loss
