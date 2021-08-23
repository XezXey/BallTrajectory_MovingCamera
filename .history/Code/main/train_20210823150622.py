from __future__ import print_function
# Import libs
import numpy as np
from numpy.lib.arraypad import pad
import torch as pt
import matplotlib.pyplot as plt
import os, sys, glob
import argparse
import time
import yaml
import shutil
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
parser.add_argument('--trajectory_type', dest='trajectory_type', type=str, help='Type of trajectory(Rolling, Projectile, MagnusProjectile)', default='Projectile')
parser.add_argument('--env', dest='env', help='Environment', default='unity')

# Wandb
parser.add_argument('--wandb_dir', help='Path to WanDB directory', type=str, default='./')
parser.add_argument('--wandb_name', dest='wandb_name', type=str, help='WanDB session name', default=None)
parser.add_argument('--wandb_notes', dest='wandb_notes', type=str, help='WanDB notes', default='')
parser.add_argument('--wandb_tags', dest='wandb_tags', type=str, help='WanDB tags name', default=None)
parser.add_argument('--wandb_mode', dest='wandb_mode', type=str, help='WanDB mode (run/dryrun)', default='run')

# ckpt
parser.add_argument('--save_ckpt', dest='save_ckpt', type=str, help='Path to save a model ckpt', default=None)
parser.add_argument('--load_ckpt', dest='load_ckpt', type=str, help='Path to load a trained model ckpt', default=None)

# Training parameters
parser.add_argument('--batch_size', dest='batch_size', type=int, help='Samples in batch', default=512)
parser.add_argument('--n_epochs', dest='n_epochs', type=int, help='Train for #n epoch', default=100000)
parser.add_argument('--lr', help='Learning rate', type=float, default=0.001)
parser.add_argument('--clip', dest='clip', type=float, help='Clipping gradients value', default=10)
parser.add_argument('--decay_gamma', help='Gamma (Decay rate)', type=float, default=0.7)
parser.add_argument('--decay_cycle', help='Decay cycle', type=int, default=50)
parser.add_argument('--canonicalize', dest='canonicalize', default=None)
parser.add_argument('--noise', dest='noise', help='Noise on the fly', action='store_true')
parser.add_argument('--no_noise', dest='noise', help='Noise on the fly', action='store_false')
parser.add_argument('--noise_sd', dest='noise_sd', help='Std. of noise', type=float, default=None)
parser.add_argument('--annealing', dest='annealing', help='Apply annealing', action='store_true', default=False)
parser.add_argument('--annealing_cycle', dest='annealing_cycle', type=int, help='Apply annealing every n epochs', default=5)
parser.add_argument('--annealing_gamma', dest='annealing_gamma', type=float, help='Apply annealing every n epochs', default=0.95)

# Visualization
parser.add_argument('--visualize', dest='visualize', help='Visualize the trajectory', action='store_true', default=False)
parser.add_argument('--visualization_path', dest='visualization_path', type=str, help='Path to visualization directory', default='./visualize_html/')

# Weighted combining
parser.add_argument('--si_pred_ramp', help='Directional prediction with ramp weight', action='store_true', default=False)

# Model
parser.add_argument('--pipeline', dest='pipeline', help='Pipeline', nargs='+', default=None)

# Refinement stuff
#parser.add_argument('--in_refine', dest='in_refine', help='Input for refinement network', default='xyz')
#parser.add_argument('--out_refine', dest='out_refine', help='Output for refinement network', default='xyz')
#parser.add_argument('--n_refinement', dest='n_refinement', help='Refinement Iterations', type=int, default=1)
#parser.add_argument('--fix_refinement', dest='fix_refinement', help='Fix Refinement for 1st and last points', action='store_true', default=False)
#parser.add_argument('--refine', dest='refine', help='I/O for refinement network', default='position')

# Loss
parser.add_argument('--multiview_loss', dest='multiview_loss', help='Use Multiview loss', nargs='+', default=[])

# Features
parser.add_argument('--selected_features', dest='selected_features', help='Specify the selected features columns(eot, og, ', nargs='+', default=[])
parser.add_argument('--i_s', dest='i_s', help='input space', type=str, default='t')
parser.add_argument('--o_s', dest='o_s', help='output space', type=str, default='t')
parser.add_argument('--sc', dest='sc', help='Sin/Cos of the angle', type=str, default='def')

# Miscellaneous
parser.add_argument('--cuda_device_num', dest='cuda_device_num', type=int, help='Provide cuda device number', default=0)
parser.add_argument('--test')

# YAML-Config
parser.add_argument('--config_yaml', dest='config_yaml', type=str, help='Config parameters file', required=True)

args = parser.parse_args()

args = utils_func.yaml_to_args(args=args)

print("*"*100)
print(vars(args))

# Share args to every modules
utils_func.share_args(args)
utils_transform.share_args(args)
utils_model.share_args(args)
utils_loss.share_args(args)

# GPU initialization
if pt.cuda.is_available():
  pt.cuda.set_device(args.cuda_device_num)
  device = pt.device('cuda')
  print('[%]GPU Enabled')
else:
  device = pt.device('cpu')
  print('[%]GPU Disabled, CPU Enabled')

# Init wandb
if args.wandb_notes is None:
  args.wandb_notes = args.wandb_name

os.environ['WANDB_MODE'] = args.wandb_mode
wandb.init(project="ball-trajectory-estimation", name=args.wandb_name, tags=args.wandb_tags, notes=args.wandb_notes, dir=args.wandb_dir)

# Get selected features to input into a network
features = ['x', 'y', 'z', 'u', 'v', 'd', 'intr_x', 'intr_y', 'intr_z', 'ray_x', 'ray_y', 'ray_z', 'cam_x', 'cam_y', 'cam_z', 
            'eot', 'og', 'rad', 'f_sin', 'f_cos', 'fx', 'fy', 'fz', 'fx_norm', 'fy_norm', 'fz_norm',
            'intrinsic', 'extrinsic', 'azimuth', 'elevation', 'extrinsic_inv', 'g']
x, y, z, u, v, d, intr_x, intr_y, intr_z, ray_x, ray_y, ray_z, cam_x, cam_y, cam_z, eot, og, rad, f_sin, f_cos, fx, fy, fz, fx_norm, fy_norm, fz_norm, intrinsic, extrinsic, azimuth, elevation, extrinsic_inv, g = range(len(features))
input_col, gt_col, cpos_col, features_cols = utils_func.get_selected_cols(args=args, pred='height')

def train(input_dict_train, gt_dict_train, input_dict_val, gt_dict_val, cam_dict_train, cam_dict_val, model_dict, epoch, optimizer, annealing_weight, visualization_path='./visualize_html/'):

  ####################################
  ############# Training #############
  ####################################
  utils_model.train_mode(model_dict=model_dict)

  pred_dict_train, in_train = utils_model.fw_pass(model_dict, input_dict=input_dict_train, cam_dict=cam_dict_train, gt_dict=gt_dict_train)

  optimizer.zero_grad() # Clear existing gradients from previous epoch
  train_loss_dict, train_loss = utils_model.calculate_loss(pred_xyz=pred_dict_train['xyz'], input_dict=input_dict_train, gt_dict=gt_dict_train, pred_dict=pred_dict_train, annealing_w=annealing_weight) # Calculate the loss


  #train_loss.requires_grad_(True)
  train_loss.backward()

  # Gradient Clipping
  if args.clip > 0:
    for model in model_dict:
      pt.nn.utils.clip_grad_norm_(model_dict[model].parameters(), args.clip)

  optimizer.step()

  ####################################
  ############# Evaluate #############
  ####################################
  # Evaluating mode
  utils_model.eval_mode(model_dict=model_dict)

  pred_dict_val, in_train = utils_model.fw_pass(model_dict, input_dict=input_dict_val, cam_dict=cam_dict_val, gt_dict=gt_dict_val)

  optimizer.zero_grad() # Clear existing gradients from previous epoch
  val_loss_dict, val_loss = utils_model.calculate_loss(pred_xyz=pred_dict_val['xyz'], input_dict=input_dict_val, gt_dict=gt_dict_val, pred_dict=pred_dict_val, annealing_w=annealing_weight) # Calculate the loss

  utils_func.print_loss(loss_list=[train_loss_dict, train_loss], name='Training')
  utils_func.print_loss(loss_list=[val_loss_dict, val_loss], name='Validating')
  wandb.log({'Train Loss':train_loss.item(), 'Validation Loss':val_loss.item()})

  if args.visualize and epoch % 250 == 0:
    utils_vis.make_visualize(input_dict_train=input_dict_train, gt_dict_train=gt_dict_train, 
                              input_dict_val=input_dict_val, gt_dict_val=gt_dict_val, 
                              pred_dict_train=pred_dict_train, pred_dict_val=pred_dict_val, 
                              visualization_path=visualization_path, pred='height')

  return train_loss.item(), val_loss.item(), model_dict

def collate_fn_padd(batch):
    # Padding batch of variable length
    # Columns convention : (x, y, z, u, v, d, eot, og, rad)
    padding_value = -1000.0
    ## Get sequence lengths
    lengths = pt.tensor([trajectory.shape[0] for trajectory in batch])
    # Input features : columns 4-5 contain u, v in screen space
    ## Padding
    input_batch = [pt.Tensor(trajectory[:, input_col].astype(np.float64)) for trajectory in batch] # (4, 5, -2) = (u, v ,end_of_trajectory)
    input_batch = pad_sequence(input_batch, batch_first=True, padding_value=padding_value)
    ## Compute mask
    input_mask = (input_batch != padding_value)

    # Output features : columns 6 cotain depth from camera to projected screen
    ## Padding
    gt_batch = [pt.Tensor(trajectory[:, gt_col].astype(np.float64)) for trajectory in batch]
    gt_batch = pad_sequence(gt_batch, batch_first=True, padding_value=padding_value)
    ## Compute mask
    gt_mask = (gt_batch != padding_value)

    # Extra columns : columns that contains information for recon/auxiliary
    ## Padding
    cpos_batch = [pt.Tensor(trajectory[:, cpos_col].astype(np.float64)) for trajectory in batch] 
    cpos_batch = pad_sequence(cpos_batch, batch_first=True, padding_value=0)

    max_len = pt.max(lengths)
    # Intrinsic/Extrinsic columns
    I = [pt.Tensor(utils_transform.IE_array(trajectory, col=intrinsic)) for trajectory in batch]
    E = [pt.Tensor(utils_transform.IE_array(trajectory, col=extrinsic)) for trajectory in batch]
    E_inv = [pt.Tensor(utils_transform.IE_array(trajectory, col=extrinsic_inv)) for trajectory in batch]
    # Manually pad with eye to prevent non-invertible matrix
    for i in range(len(lengths)):
      pad_len = max_len - lengths[i]
      if pad_len == 0:
        continue
      else:
        pad_mat = pt.stack(pad_len * [pt.eye(4)])
        I[i] = pt.cat((I[i], pad_mat), dim=0)
        E[i] = pt.cat((E[i], pad_mat), dim=0)
        E_inv[i] = pt.cat((E_inv[i], pad_mat), dim=0)
    I = pt.stack(I)
    E = pt.stack(E)
    E_inv = pt.stack(E_inv)

    # Tracking
    tracking = [pt.Tensor(trajectory[:, [u, v]].astype(np.float64)) for trajectory in batch]
    tracking = pad_sequence(tracking, batch_first=True, padding_value=1)

    return {'input':[input_batch, lengths, input_mask],
            'gt':[gt_batch, lengths, gt_mask],
            'cpos':[cpos_batch],
            'tracking':[tracking],
            'I':[I], 'E':[E], 'E_inv':[E_inv]}


if __name__ == '__main__':
  print('[#]Training : Trajectory Estimation')

  # Initialize folder
  utils_func.initialize_folder(args.visualization_path)
  save_ckpt = '{}/{}/'.format(args.save_ckpt + args.wandb_tags.replace('/', '_'), args.wandb_name)
  utils_func.initialize_folder(save_ckpt)

  # Create Datasetloader for train and validation
  dataset_train = TrajectoryDataset(dataset_path=args.dataset_train_path, trajectory_type=args.trajectory_type)
  dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=10, shuffle=True, collate_fn=collate_fn_padd, pin_memory=True, drop_last=True)
  # Create Datasetloader for validation
  dataset_val = TrajectoryDataset(dataset_path=args.dataset_val_path, trajectory_type=args.trajectory_type)
  dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=10, shuffle=True, collate_fn=collate_fn_padd, pin_memory=True, drop_last=True)

  #utils_func.show_dataset_info(dataloader_train, 'Train')
  #utils_func.show_dataset_info(dataloader_val, 'Val')

  # Model definition
  min_val_loss = 2e10
  annealing_step = 0
  annealing_weight = 1.0
  annealing_weight_list = np.linspace(start=0, stop=1, num=15)
  model_dict, model_cfg = utils_func.get_model(args=args)

  model_dict = {model:model_dict[model].to(device) for model in model_dict.keys()}

  # Define optimizer, learning rate, decay and scheduler parameters
  # for model
  params = []
  for model in model_dict.keys():
    params += list(model_dict[model].parameters())
  optimizer = pt.optim.Adam(params, lr=args.lr)
  lr_scheduler = pt.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.decay_gamma)
  start_epoch = 1

  # Load the ckpt if it's available.
  if args.load_ckpt is None:
    # Create a model
    print('===>No model ckpt')
    print('[#] Define the Learning rate, Optimizer, Decay rate and Scheduler...')
    # Save the config.yaml
    shutil.copy2(src=args.config_yaml, dst='{}/config.yaml'.format(save_ckpt))
  else:
    print('===>Load ckpt with Optimizer state, Decay and Scheduler state')
    print('[#] Loading ... {}'.format(args.load_ckpt))
    model_dict, optimizer, start_epoch, lr_scheduler, min_val_loss, annealing_scheduler = utils_func.load_ckpt_train(model_dict, optimizer, lr_scheduler)
    annealing_step = annealing_scheduler['step']
    annealing_weight = annealing_scheduler['weight']

  print('[#]Model Architecture')
  for model in model_cfg.keys():
    print('####### Model - {} #######'.format(model))
    print(model_dict[model])
    for name, param in model_dict[model].named_parameters():
      print(name, param.shape)
    # exit()
    # Log metrics with wandb
    wandb.watch(model_dict[model])

  # Training settings
  trajectory_val_iterloader = iter(dataloader_val)
  for epoch in range(start_epoch, args.n_epochs+1):
    accumulate_train_loss = []
    accumulate_val_loss = []
    # Fetch the Validation set (Get each batch for each training epochs)
    utils_func.random_seed()
    try:
      batch_val = next(trajectory_val_iterloader)
    except StopIteration:
      trajectory_val_iterloader = iter(dataloader_val)
      batch_val = next(trajectory_val_iterloader)

    input_dict_val = {'input':batch_val['input'][0].to(device), 'lengths':batch_val['input'][1].to(device), 'mask':batch_val['input'][2].to(device)}
    gt_dict_val = {'gt':batch_val['gt'][0].to(device), 'lengths':batch_val['gt'][1].to(device), 'mask':batch_val['gt'][2].to(device)}
    cam_dict_val = {'I':batch_val['I'][0].to(device), 'E':batch_val['E'][0].to(device), 'E_inv':batch_val['E_inv'][0].to(device),
                    'tracking':batch_val['tracking'][0], 'cpos':batch_val['cpos'][0].to(device)}

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>[Epoch : {}/{}]<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(epoch, args.n_epochs))

    # Log the learning rate
    for param_group in optimizer.param_groups:
      print("[#]Learning rate : ", param_group['lr'])
      wandb.log({'Learning Rate':param_group['lr'], 'n_epochs':epoch})

    # Training a model iterate over dataloader to get each batch and pass to train function
    for batch_idx, batch_train in enumerate(dataloader_train):
      print('===> [Minibatch {}/{}].........'.format(batch_idx+1, len(dataloader_train)), end='\n')
      # Training set (Each index in batch_train came from the collate_fn_padd)
      input_dict_train = {'input':batch_train['input'][0].to(device), 'lengths':batch_train['input'][1].to(device), 'mask':batch_train['input'][2].to(device)}
      gt_dict_train = {'gt':batch_train['gt'][0].to(device), 'lengths':batch_train['gt'][1].to(device), 'mask':batch_train['gt'][2].to(device)}
      cam_dict_train = {'I':batch_train['I'][0].to(device), 'E':batch_train['E'][0].to(device), 'E_inv':batch_train['E_inv'][0].to(device),
                    'tracking':batch_train['tracking'][0], 'cpos':batch_train['cpos'][0].to(device)}


      # Call function to train
      train_loss, val_loss, model_dict = train(input_dict_train=input_dict_train, gt_dict_train=gt_dict_train, cam_dict_train=cam_dict_train,
                                              input_dict_val=input_dict_val, gt_dict_val=gt_dict_val, cam_dict_val=cam_dict_val, 
                                              annealing_weight=annealing_weight, model_dict=model_dict, optimizer=optimizer,
                                              epoch=epoch, visualization_path=args.visualization_path)

      accumulate_val_loss.append(val_loss)
      accumulate_train_loss.append(train_loss)

    # Get the average loss for each epoch over entire dataset
    val_loss_per_epoch = np.mean(accumulate_val_loss)
    train_loss_per_epoch = np.mean(accumulate_train_loss)
    # Log the each epoch loss
    wandb.log({'Epoch Train Loss':train_loss_per_epoch, 'Epoch Validation Loss':val_loss_per_epoch})

    # Decrease learning rate every n_epochs % decay_cycle batch
    if epoch % args.decay_cycle == 0:
      lr_scheduler.step()
      for param_group in optimizer.param_groups:
        print("[#]Stepping Learning rate to ", param_group['lr'])

    # Decrease learning rate every n_epochs % annealing_cycle batch
    if epoch % args.annealing_cycle == 0:
      # annealing_weight = annealing_weight * args.annealing_gamma ** annealing_step
      if annealing_step < len(annealing_weight_list):
        annealing_weight = annealing_weight_list[annealing_step]
      else:
        annealing_weight = annealing_weight_list[-1]
      annealing_step += 1
      print("[#]Stepping annealing weight to ", annealing_weight)

    print('[#]Finish Epoch : {}/{}.........Train loss : {:.3f}, Val loss : {:.3f}'.format(epoch, args.n_epochs, train_loss_per_epoch, val_loss_per_epoch))

    # Save the model ckpt every finished the epochs
    if min_val_loss > val_loss_per_epoch:
      # Save model ckpt
      save_ckpt_best = '{}/{}_best.pth'.format(save_ckpt, args.wandb_name)
      print('[+++]Saving the best model ckpt : Prev loss {:.3f} > Curr loss {:.3f}'.format(min_val_loss, val_loss_per_epoch))
      print('[+++]Saving the best model ckpt to : ', save_ckpt_best)
      min_val_loss = val_loss_per_epoch
      annealing_scheduler = {'step':annealing_step, 'weight':annealing_weight}
      # Save to directory
      ckpt = {'epoch':epoch+1, 'optimizer':optimizer.state_dict(), 'lr_scheduler':lr_scheduler.state_dict(), 'min_val_loss':min_val_loss, 'model_cfg':model_cfg, 'annealing_scheduler':annealing_scheduler}
      for model in model_cfg:
        ckpt[model] = model_dict[model].state_dict()
      pt.save(ckpt, save_ckpt_best)
      pt.save(ckpt, os.path.join(wandb.run.dir, 'ckpt_best.pth'))
      

    else:
      print('[#]Not saving the best model ckpt : Val loss {:.3f} not improved from {:.3f}'.format(val_loss_per_epoch, min_val_loss))


    if epoch % 20 == 0:
      # Save the lastest ckpt for continue training every 10 epoch
      save_ckpt_lastest = '{}/{}_lastest.pth'.format(save_ckpt, args.wandb_name)
      print('[#]Saving the lastest ckpt to : ', save_ckpt_lastest)
      ckpt = {'epoch':epoch+1, 'optimizer':optimizer.state_dict(), 'lr_scheduler':lr_scheduler.state_dict(), 'min_val_loss':min_val_loss, 'model_cfg':model_cfg, 'annealing_scheduler':annealing_scheduler}
      for model in model_cfg:
        ckpt[model] = model_dict[model].state_dict()
      pt.save(ckpt, save_ckpt_lastest)
      pt.save(ckpt, os.path.join(wandb.run.dir, 'ckpt_lastest.pth'))

  print("[#] Done")
