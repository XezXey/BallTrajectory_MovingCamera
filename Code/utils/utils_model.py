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
      print(ray_cl[0][:10])
      print(in_f_cl[0][:10])
      in_f_decl, _, _ = utils_transform.canonicalize(pts=in_f_cl[..., [0, 1, 2]], E=cam_dict['E'], deg=angle)
      ray_decl, _, _ = utils_transform.canonicalize(pts=ray_cl[..., [0, 1, 2]], E=cam_dict['E'], deg=angle)
      print(ray[0][:10] - ray_decl[0][:10])
      print(in_f[0][:10, [0, 1, 2]] - in_f_decl[0][:10])
      exit()


    if 'height' in args.pipeline:

      h = output_space()

      

def prep_input(in_f):
  '''
  Prepare input to have correct space(t, dt, tdt), sin-cos
  Input : 
    1. in_f : input features in shape (batch, seq_len, 5)
  Output : 
    1. in_f : input features after change into specified space(t, dt, tdt) and sin-cos
  '''
  pass

    

def input_space():
  '''
  '''
  pass

def output_space():
  '''
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