from __future__ import print_function
# Import libs
import numpy as np
np.set_printoptions(precision=4)
import torch as pt
pt.manual_seed(25)
import glob
import os
import argparse
import sys
import time
sys.path.append(os.path.realpath('../'))
from tqdm import tqdm
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import PIL.Image
import io
from sklearn.metrics import confusion_matrix
import wandb
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from functools import partial
# Utils
from utils.dataloader import TrajectoryDataset
import utils.utils_func as utils_func
import utils.utils_vis as utils_vis
import utils.transformation as utils_transform
import utils.utils_model as utils_model

# Loss
import utils.loss as utils_loss

# Argumentparser for input
parser = argparse.ArgumentParser(description='Predict the 3D Trajectory')

# Datapath
parser.add_argument('--dataset_train_path', dest='dataset_train_path', type=str, help='Path to training set', default=None)
parser.add_argument('--dataset_val_path', dest='dataset_val_path', type=str, help='Path to validation set', default=None)
parser.add_argument('--dataset_test_path', dest='dataset_test_path', type=str, help='Path to test set', default=None)
parser.add_argument('--trajectory_type', dest='trajectory_type', type=str, help='Type of trajectory(Rolling, Projectile, MagnusProjectile)', default=None)
parser.add_argument('--env', dest='env', help='Environment', default=None)

# Wandb
parser.add_argument('--wandb_dir', help='Path to WanDB directory', type=str, default=None)
parser.add_argument('--wandb_name', dest='wandb_name', type=str, help='WanDB session name', default=None)
parser.add_argument('--wandb_notes', dest='wandb_notes', type=str, help='WanDB notes', default='')
parser.add_argument('--wandb_tags', dest='wandb_tags', type=str, help='WanDB tags name', default=None)
parser.add_argument('--wandb_mode', dest='wandb_mode', type=str, help='WanDB mode (run/dryrun)', default=None)

# ckpt
parser.add_argument('--save_ckpt', dest='save_ckpt', type=str, help='Path to save a model ckpt', default=None)
parser.add_argument('--load_ckpt', dest='load_ckpt', type=str, help='Path to load a trained model ckpt', default=None)

# Training parameters
parser.add_argument('--batch_size', dest='batch_size', type=int, help='Samples in batch', default=None)
parser.add_argument('--n_epochs', dest='n_epochs', type=int, help='Train for #n epoch', default=None)
parser.add_argument('--lr', help='Learning rate', type=float, default=None)
parser.add_argument('--clip', dest='clip', type=float, help='Clipping gradients value', default=None)
parser.add_argument('--decay_gamma', help='Gamma (Decay rate)', type=float, default=None)
parser.add_argument('--decay_cycle', help='Decay cycle', type=int, default=None)
parser.add_argument('--canonicalize', dest='canonicalize', default=None)

## Annealing
parser.add_argument('--annealing', dest='annealing', help='Apply annealing', action='store_true', default=None)
parser.add_argument('--no_annealing', dest='annealing', help='Apply annealing', action='store_false', default=None)
parser.add_argument('--annealing_cycle', dest='annealing_cycle', type=int, help='Apply annealing every n epochs', default=None)
parser.add_argument('--annealing_gamma', dest='annealing_gamma', type=float, help='Apply annealing every n epochs', default=None)
parser.add_argument('--recon', dest='recon', type=str, help='Reconstruction selection (noisy/clean)', default=None)

## Noise
parser.add_argument('--noise', dest='noise', help='Noise on the fly', action='store_true', default=None)
parser.add_argument('--no_noise', dest='noise', help='Noise on the fly', action='store_false', default=None)

## Augment
parser.add_argument('--augment', dest='augment', help='Apply an augmented training', action='store_true', default=None)
parser.add_argument('--no_augment', dest='augment', help='Apply an augmented training', action='store_false', default=None)
parser.add_argument('--augment_perc', dest='augment_perc', help='Augmented percentage length', type=float, default=1.0)
parser.add_argument('--augment_start', dest='augment_start', help='Augmented start', type=int, default=0)

# Optimization
parser.add_argument('--optim_init_h', dest='optim_init_h', help='Optimize for initial height', action='store_true', default=None)
parser.add_argument('--optim_latent', dest='optim_latent', help='Optimize for latent', action='store_true', default=None)
parser.add_argument('--optim_analyse', dest='optim_analyse', help='Latent optimization analyse', type=str, default=None)

# Visualization
parser.add_argument('--visualize', dest='visualize', help='Visualize the trajectory', action='store_true', default=None)
parser.add_argument('--vis_path', dest='vis_path', type=str, help='Path to visualization directory', default=None)

# Input Variation
parser.add_argument('--input_variation', dest='input_variation', type=str, help='Input features variation', default=None)

# Model
parser.add_argument('--pipeline', dest='pipeline', help='Pipeline', nargs='+', default=None)

# Loss
parser.add_argument('--multiview_loss', dest='multiview_loss', help='Use Multiview loss', nargs='+', default=None)

# Features
parser.add_argument('--selected_features', dest='selected_features', help='Specify the selected features columns(eot, og, ', nargs='+', default=None)
parser.add_argument('--sc', dest='sc', help='Sin/Cos of the angle', type=str, default=None)

# Miscellaneous
parser.add_argument('--cuda_device_num', dest='cuda_device_num', type=int, help='Provide cuda device number', default=0)
parser.add_argument('--save_cam_traj', dest='save_cam_traj', type=str, help='Save a trajectory', default=None)
parser.add_argument('--save_suffix', dest='save_suffix', type=str, help='Save file suffix', default='')
parser.add_argument('--no_gt', dest='no_gt', help='Is ground-truth available?', action='store_true', default=False)
parser.add_argument('--w', dest='w', type=float, help='width', default=None)
parser.add_argument('--h', dest='h', type=float, help='height', default=None)
parser.add_argument('--fps', dest='fps', type=float, help='fps', default=None)

# YAML-Config
parser.add_argument('--config_yaml', dest='config_yaml', type=str, help='Config parameters file', required=True)

args = parser.parse_args()
args = utils_func.yaml_to_args(args=args)
print("*"*100)

# Share args to every modules
utils_func.share_args(args)
utils_transform.share_args(args)
utils_model.share_args(args)
utils_loss.share_args(args)
utils_vis.share_args(args)

# GPU initialization
if pt.cuda.is_available():
  pt.cuda.set_device(args.cuda_device_num)
  device = pt.device('cuda')
  print('[%]GPU Enabled')
else:
  device = pt.device('cpu')
  print('[%]GPU Disabled, CPU Enabled')

# Get selected features to input into a network
# Not og features
"""
features = ['x', 'y', 'z', 'u', 'v', 'd', 'intr_x', 'intr_y', 'intr_z', 'ray_x', 'ray_y', 'ray_z', 
            'eot', 'cd', 'rad', 'f_sin', 'f_cos', 'fx', 'fy', 'fz', 'fx_norm', 'fy_norm', 'fz_norm',
            'intrinsic', 'extrinsic', 'azimuth', 'elevation', 'extrinsic_inv', 'g']
x, y, z, u, v, d, intr_x, intr_y, intr_z, ray_x, ray_y, ray_z, eot, cd, rad, f_sin, f_cos, fx, fy, fz, fx_norm, fy_norm, fz_norm, intrinsic, extrinsic, azimuth, elevation, extrinsic_inv, g = range(len(features))
"""

features = ['x', 'y', 'z', 'u', 'v', 'd', 'intr_x', 'intr_y', 'intr_z', 'ray_x', 'ray_y', 'ray_z', 
            'eot', 'cd', 'og', 'hw', 'rad', 'f_sin', 'f_cos', 'fx', 'fy', 'fz', 'fx_norm', 'fy_norm', 'fz_norm',
            'intrinsic', 'extrinsic', 'azimuth', 'elevation', 'extrinsic_inv', 'g']
x, y, z, u, v, d, intr_x, intr_y, intr_z, ray_x, ray_y, ray_z, eot, cd, og, hw, rad, f_sin, f_cos, fx, fy, fz, fx_norm, fy_norm, fz_norm, intrinsic, extrinsic, azimuth, elevation, extrinsic_inv, g = range(len(features))
input_col, gt_col, features_col = utils_func.get_selected_cols(args=args, pred='height')

def concat_traj(recon_traj_all, space, concat=True):
  '''
  Concat all trajectory for evaluate
  1. Input: 
    - recon_traj_all : list of dict contains
        - xyz
        - gt
        - xyz_refined
        - seq_len
  '''
  trajectory = {k:[] for k in space}
  for s in space:
    for i in range(len(recon_traj_all)):
      # Each batch
      tmp = recon_traj_all[i][s]
      seq_len = recon_traj_all[i]['seq_len']
      for j in range(recon_traj_all[i][s].shape[0]):
        # Each trajectory
        trajectory[s].append(tmp[j][:seq_len[j]])
    if concat:
      trajectory[s] = np.concatenate(trajectory[s], axis=0)
  return trajectory

def evaluate(recon_traj_all):
  print("*"*100)
  print("[#]Evaluation Results : \"{}/{}\"".format(args.wandb_tags, args.wandb_name))
  print("[#]Data : \"{}\"".format(args.dataset_test_path))
  print("*"*100)

  distance = ['MAE', 'MSE', 'RMSE', 'RMSE-DISTANCE']
  space = ['gt', 'xyz', 'xyz_refined'] if 'refinement' in args.pipeline else ['gt', 'xyz']
  trajectory = concat_traj(recon_traj_all, space)


  for s in space:
    if s == 'gt':
      continue
    print("Space : ", s)
    gt = trajectory['gt']
    pred = trajectory[s]
    for e_d in distance:
      print("===> Distance : ", e_d)
      if e_d == 'MAE':
        err = np.abs(gt - pred)
        mean = np.mean(err, axis=0)
        std = np.std(err, axis=0)
        print("\tMEAN : ", mean)
        print("\tSD : ", std)
      elif e_d == 'MSE':
        err = (gt - pred)**2
        mean = np.mean(err, axis=0)
        std = np.std(err, axis=0)
        print("\tMEAN : ", mean)
        print("\tSD : ", std)
      elif e_d == 'RMSE':
        err = (gt - pred)**2
        rmse = np.sqrt(np.mean(err, axis=0))
        print("\tRMSE : ", rmse)
      elif e_d == 'RMSE-DISTANCE' and (s == 'xyz' or s == 'xyz_refined'):
        rmse_distance_1 = np.mean(np.sqrt(np.sum(((gt - pred)**2), axis=-1)), axis=0)
        print("\tRMSE-DISTANCE-1 : ", rmse_distance_1)
        rmse_distance_2 = np.sqrt(np.mean(np.sum(((gt - pred)**2), axis=-1), axis=0))
        print("\tRMSE-DISTANCE-2 : ", rmse_distance_2)

    print("*"*100)
  return trajectory

def predict(input_dict_test, gt_dict_test, cam_dict_test, model_dict, tid, threshold=0.01):
  # Random noise
  noise_sd = args.pipeline['height']['noise_sd']
  if type(noise_sd) == int:
    noise_sd = float(noise_sd)
  elif type(noise_sd) == str:
    l, h = int(noise_sd.split('t')[0]), int(noise_sd.split('t')[-1])
    noise_sd = np.random.uniform(low=l, high=h)
  args.pipeline['height']['noise_sd_'] = noise_sd

  ####################################
  ############# Testing ##############
  ####################################
  pred_dict_test = {}
  # Evaluating mode
  utils_model.eval_mode(model_dict=model_dict)

  latent_dict_test = {module:None for module in args.pipeline}
  if (args.optim_init_h or args.optim_latent) and (args.optim_analyse is not None):
    pred_dict_test, in_test, loss_landscape = utils_model.fw_pass_optim_analyse(model_dict, input_dict=input_dict_test, cam_dict=cam_dict_test, gt_dict=gt_dict_test, latent_dict=latent_dict_test, set_='test', tid=tid)
    utils_func.save_gridsearch(loss_landscape, gt_dict=gt_dict_test, cam_dict=cam_dict_test, tid=tid)
  elif (args.optim_init_h or args.optim_latent) and (args.optim_analyse is None):
    pred_dict_test, in_test, latent_dict_test = utils_model.fw_pass_optim(model_dict, input_dict=input_dict_test, cam_dict=cam_dict_test, gt_dict=gt_dict_test, latent_dict=latent_dict_test, set_='test')
  else:
    pred_dict_test, in_test = utils_model.fw_pass(model_dict, input_dict=input_dict_test, cam_dict=cam_dict_test, gt_dict=gt_dict_test, latent_dict=latent_dict_test, set_='test')

  ###################################
  ############ Evaluation ###########
  ###################################
  # Calculate loss per trajectory
  recon_traj = {'gt':gt_dict_test['gt'][..., [0, 1, 2]].detach().cpu().numpy(), 
                'flag':pred_dict_test['flag'].detach().cpu().numpy() if 'flag' in args.pipeline else None,
                'xyz':pred_dict_test['xyz'].detach().cpu().numpy(), 
                'xyz_refined':pred_dict_test['xyz_refined'].detach().cpu().numpy() if 'refinement' in args.pipeline else None,
                'seq_len':gt_dict_test['lengths'].detach().cpu().numpy(), 
                'cam_dict':cam_dict_test}

  #utils_func.print_loss(loss_list=[test_loss_dict, test_loss], name='Testing')

  if args.visualize:
    utils_vis.inference_vis(input_dict=input_dict_test, pred_dict=pred_dict_test, gt_dict=gt_dict_test, 
                            cam_dict=cam_dict_test, latent_dict=latent_dict_test)

  return recon_traj

def collate_fn_padd(batch, set_):
  if 'unity' not in args.env:
    global u, v, intrinsic, extrinsic, extrinsic_inv, x, y, z
    u, v, intrinsic, extrinsic, extrinsic_inv, x, y, z = 0, 1, 2, 3, 4, 5, 6, 7
    global input_col, gt_col, features_col
    input_col = [u, v]
    gt_col = [x, y, z]
    features_col = []

  # Padding batch of variable length
  if args.augment:
    batch = utils_func.augment_pred(batch=batch)

  padding_value = -1000.0
  ## Get sequence lengths
  lengths = pt.tensor([trajectory.shape[0] for trajectory in batch])
  # Input features : (u, v)
  #input_batch = [pt.Tensor(np.flip(trajectory[:, input_col].astype(np.float64), axis=0).copy()) for trajectory in batch]
  input_batch = [pt.Tensor(trajectory[:, input_col].astype(np.float64)) for trajectory in batch]
  input_batch = pad_sequence(input_batch, batch_first=True, padding_value=padding_value)
  ## Compute mask
  input_mask = (input_batch != padding_value)
  input_mask = input_mask[..., [0]].repeat(1, 1, 3)

  # Auxiliary features : e.g. eot, cd, rad, f_sin, f_cos, ..., etc.
  ## Padding
  features_batch = [pt.Tensor(trajectory[:, features_col].astype(np.float64)) for trajectory in batch]
  features_batch = pad_sequence(features_batch, batch_first=True, padding_value=padding_value)

  # Output features : (x, y, z)
  if args.no_gt:
    gt_batch = None
    gt_mask = None
  else:
    gt_batch = [pt.Tensor(trajectory[:, gt_col].astype(np.float64)) for trajectory in batch]
    gt_batch = pad_sequence(gt_batch, batch_first=True, padding_value=padding_value)
    ## Compute mask
    gt_mask = (gt_batch != padding_value)

  max_len = pt.max(lengths)
  # Intrinsic/Extrinsic columns
  I = [pt.Tensor(utils_transform.IE_array(trajectory, col=intrinsic)) for trajectory in batch]
  E = [pt.Tensor(utils_transform.IE_array(trajectory, col=extrinsic)) for trajectory in batch]
  Einv = [pt.Tensor(utils_transform.IE_array(trajectory, col=extrinsic_inv)) for trajectory in batch]
  cpos_batch = []
  # Manually pad with eye to prevent non-invertible matrix
  for i in range(len(lengths)):
    pad_len = max_len - lengths[i]
    if pad_len == 0:
      cpos_batch.append(Einv[i][:, :3, -1])
    else:
      pad_mat = pt.stack(pad_len * [pt.eye(4)])
      I[i] = pt.cat((I[i], pad_mat), dim=0)
      E[i] = pt.cat((E[i], pad_mat), dim=0)
      Einv[i] = pt.cat((Einv[i], pad_mat), dim=0)
      cpos_batch.append(Einv[i][:, :3, -1])

  I = pt.stack(I)
  E = pt.stack(E)
  Einv = pt.stack(Einv)
  cpos_batch = pt.stack(cpos_batch)

  # Tracking
  tracking = [pt.Tensor(trajectory[:, [u, v]].astype(np.float64)) for trajectory in batch]
  tracking = pad_sequence(tracking, batch_first=True, padding_value=1)

  return {'input':[input_batch, features_batch, lengths, input_mask],
          'gt':[gt_batch, lengths, gt_mask],
          'cpos':[cpos_batch],
          'tracking':[tracking],
          'I':[I], 'E':[E], 'Einv':[Einv]}

if __name__ == '__main__':
  print('[#]Testing : Trajectory Estimation')

  # Initialize folder
  utils_func.initialize_folder(args.vis_path)

  # Create Datasetloader for test and testidation
  dataset_test = TrajectoryDataset(dataset_path=args.dataset_test_path, trajectory_type=args.trajectory_type)
  dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=10, shuffle=False, collate_fn=partial(collate_fn_padd, set_='test'), pin_memory=True, drop_last=False)
  # Cast it to iterable object
  trajectory_test_iterloader = iter(dataloader_test)

  #utils_func.show_dataset_info(dataloader_test, 'Test')

  # Model definition
  model_dict, model_cfg = utils_func.get_model(args=args)
  model_dict = {model:model_dict[model].to(device) for model in model_dict.keys()}

  # Load the ckpt if it's available.
  ckpt = ['best', 'lastest', 'best_traj_ma']
  if (args.load_ckpt is None) or (args.load_ckpt not in ckpt):
    # Create a model
    raise ValueError('===>No model ckpt')
  else:
    print('===>Load ckpt with Optimizer state, Decay and Scheduler state')
    ckpt = '{}/{}/{}/{}_{}.pth'.format(args.save_ckpt, args.wandb_tags, args.wandb_name, args.wandb_name, args.load_ckpt)
    print('[#] Loading ... {}'.format(ckpt))
    model_dict = utils_func.load_ckpt_predict(model_dict, ckpt)

  print('[#]Model Architecture')
  for model in model_cfg.keys():
    print('####### Model - {} #######'.format(model))
    print(model_dict[model])

  # Test a model iterate over dataloader to get each batch and pass to predict function
  run_time = []
  recon_traj_all = []
  all_batch_trajectory = {'gt_xyz':[], 'pred_xyz':[], 'gt_h':[], 'pred_h':[]}
  n_trajectory = 0
  for batch_idx, batch_test in tqdm(enumerate(dataloader_test), disable=True):
    print("[#]Batch-{}".format(batch_idx))
    if batch_idx in [27, 32, 33, 34, 35, 36] + list(range(39, 61)) and args.env == 'tennis':
      continue
    if batch_idx >= 30 and args.env == 'mocap':
      break

    input_dict_test = {'input':batch_test['input'][0].to(device), 'aux':batch_test['input'][1].to(device), 'lengths':batch_test['input'][2].to(device), 'mask':batch_test['input'][3].to(device)}
    if args.no_gt:
      gt_dict_test = None
    else:
      gt_dict_test = {'gt':batch_test['gt'][0].to(device), 'lengths':batch_test['gt'][1].to(device), 'mask':batch_test['gt'][2].to(device)}
    cam_dict_test = {'I':batch_test['I'][0].to(device), 'E':batch_test['E'][0].to(device), 'Einv':batch_test['Einv'][0].to(device),
                    'tracking':batch_test['tracking'][0].to(device), 'cpos':batch_test['cpos'][0].to(device)}

    # Call function to test
    start_time = time.time()
    recon_traj = predict(input_dict_test=input_dict_test, gt_dict_test=gt_dict_test, 
                                                           cam_dict_test=cam_dict_test, model_dict=model_dict, tid=batch_idx)

    recon_traj_all.append(recon_traj)
    n_trajectory += input_dict_test['input'].shape[0]

    run_time.append(time.time()-start_time)

  _ = evaluate(recon_traj_all)

  print("[#] Runtime : {}+-{}".format(np.mean(run_time), np.std(run_time)))

  # Save prediction file
  if args.save_cam_traj is not None:
    utils_func.initialize_folder(args.save_cam_traj)
    utils_func.save_cam_traj(trajectory=recon_traj_all, n=n_trajectory)
  print("[#] Done")

