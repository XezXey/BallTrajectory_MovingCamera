# Libs
import os, sys, time

from numpy.lib.npyio import save
sys.path.append(os.path.realpath('../'))
import numpy as np
import torch as pt
import json
import plotly
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import wandb
import yaml

# Utils
import utils.transformation as utils_transform

# Models
from models.height_module import Height_Module
# Loss
import utils.loss as loss

# GPU initialization
if pt.cuda.is_available():
  device = pt.device('cuda')
else:
  device = pt.device('cpu')

args=None
features = ['x', 'y', 'z', 'u', 'v', 'd', 'intr_x', 'intr_y', 'intr_z', 'ray_x', 'ray_y', 'ray_z', 'cam_x', 'cam_y', 'cam_z', 
            'eot', 'og', 'rad', 'f_sin', 'f_cos', 'fx', 'fy', 'fz', 'fx_norm', 'fy_norm', 'fz_norm',
            'intrinsic', 'extrinsic', 'azimuth', 'elevation', 'extrinsic_inv', 'g']
x, y, z, u, v, d, intr_x, intr_y, intr_z, ray_x, ray_y, ray_z, cam_x, cam_y, cam_z, eot, og, rad, f_sin, f_cos, fx, fy, fz, fx_norm, fy_norm, fz_norm, intrinsic, extrinsic, azimuth, elevation, extrinsic_inv, g = range(len(features))

def share_args(a):
  global args
  args = a

def initialize_folder(path):
  if not os.path.exists(path):
      os.makedirs(path)

def random_seed():
  print('-' * 50)
  print("[#] Seeding...")
  print("OLD RNG : ", pt.get_rng_state())
  pt.manual_seed(time.time())
  print("NEW RNG : ", pt.get_rng_state())
  print('-' * 50)

def get_selected_cols(args, pred):
  # Flag/Extra features columns
  features_cols = []
  if 'eot' in args.selected_features:
    features_cols.append(eot)
  if 'og' in args.selected_features:
    features_cols.append(og)
  if 'rad' in args.selected_features:
    features_cols.append(rad)
  if 'f_sin' in args.selected_features:
    features_cols.append(f_sin)
  if 'f_cos' in args.selected_features:
    features_cols.append(f_cos)
  if 'fx' in args.selected_features:
    features_cols.append(fx)
  if 'fy' in args.selected_features:
    features_cols.append(fy)
  if 'fz' in args.selected_features:
    features_cols.append(fz)
  if 'fx_norm' in args.selected_features:
    features_cols.append(fx_norm)
  if 'fy_norm' in args.selected_features:
    features_cols.append(fy_norm)
  if 'fz_norm' in args.selected_features:
    features_cols.append(fz_norm)

  if pred=='height' and args.env=='unity':
    input_col = [intr_x, intr_y, intr_z, elevation, azimuth] + features_cols
    gt_col = [x, y, z] + features_cols
    cpos_col = [cam_x, cam_y, cam_z]
    #cam_params = [intrinsic, extrinsic, extrinsic_inv]

  print('='*47 + "Features" + '='*47)
  print('Prediction = {}, Environment = {}'.format(pred, args.env))
  print("Available features : ", ['{}-{}'.format(features[idx], idx) for idx in range(len(features))])
  print("Selected features : ", features_cols)
  print("1. input_col = ", input_col)
  print("2. gt_col = ", gt_col)
  print('='*100)
  return input_col, gt_col, cpos_col, features_cols

def get_extra_fsize(module):

    #############################################
    ############ EXTRA INPUT FEATURES ###########
    #############################################
    extra_in = sum(module['extra_in'])           # Extra input to module (e.g. force, acc)

    #############################################
    ########### EXTRA OUTPUT FEATURES ###########
    #############################################
    extra_out = sum(module['extra_out'])

    # Weighted Combining 
    if args.si_pred_ramp:
      # Predict only one direction, but aggregate into 2 direction
      extra_out += 1

    return extra_in, extra_out
    
def get_model(args):

  #############################################
  ############## Model I/O Size ###############
  #############################################

  model_dict = {}
  model_cfg = {}
  for module_name in args.pipeline:
    model = None
    module = args.pipeline[module_name]                 # Module
    extra_in, extra_out = get_extra_fsize(module)       # Extra input/output features
    in_node = module['in_node'] + extra_in              # Input node
    out_node = module['out_node'] + extra_out           # Output node
    arch = module['arch']                               # Architecture

    if module_name == 'height': 
      model = Height_Module(in_node=in_node, out_node=out_node, 
                    batch_size=args.batch_size, trainable_init=module['trainable_init'], 
                    is_bidirectional=module['bidirectional'], 
                    mlp_hidden=module['mlp_hidden'], mlp_stack=module['mlp_stack'],
                    rnn_hidden=module['rnn_hidden'], rnn_stack=module['rnn_stack'],)

      model_cfg[module_name] = {'in_node':in_node, 'out_node':out_node,
                      'mlp_hidden':module['mlp_hidden'], 'mlp_stack':module['mlp_stack'],
                      'rnn_hidden':module['rnn_hidden'], 'rnn_stack':module['rnn_stack'],}
    else :
      print("[#] Please input a correct pipeline...")
      exit()

    # Valid module
    if model is not None:
      model_dict[module_name] = model
    
  # Terminate if no module added
  if len(model_dict) == 0:
    print("[#] No module added...EXIT!!!")
    exit()

  return model_dict, model_cfg

def reverse_masked_seq(seq, lengths):
  '''
  Flip #n elements according to given lengths
  eg. x = [1, 2, 3, 4 , 5], len = 5
  pt.flip(x[:len], dims=[0]) ===> [5, 4, 3, 2, 1]
  '''
  seq_flip = seq.clone()
  for i in range(seq.shape[0]):
    seq_flip[i][:lengths[i]] = pt.flip(seq[i][:lengths[i], :], dims=[0])
  return seq_flip

def construct_bipred_weight(weight, lengths):
  weight_scaler = pt.nn.Sigmoid()
  weight = weight_scaler(weight)
  fw_weight = pt.cat((pt.ones(weight.shape[0], 1, 1).cuda(), weight), dim=1)
  bw_weight = pt.cat((pt.zeros(weight.shape[0], 1, 1).cuda(), 1-weight), dim=1)
  for i in range(weight.shape[0]):
    # Forward prediction weight
    fw_weight[i][lengths[i]] = 0.
    # Bachward prediction weight
    bw_weight[i][lengths[i]] = 1.
    # print(pt.cat((fw_weight, bw_weight), dim=2)[i][:lengths[i]+20])
    # exit()
  # print(weight.shape, lengths.shape)

  return pt.cat((fw_weight, bw_weight), dim=2)

def construct_w_ramp(weight_template, lengths):
  '''
  Create the ramp weight from weight template tensor
  Input : 
    1. weight_template : tensor that determined the weight shape -> (batch, seq_len, 1)
    2. lengths : legnths of each seq to construct the ramp weight and ignore the padding
  Output : 
    1. weight : shape same as weight_template
  '''

  fw_weight = pt.zeros(size=weight_template.shape).cuda()
  bw_weight = pt.zeros(size=weight_template.shape).cuda()
  for i in range(weight_template.shape[0]):
    # Forward prediction weight
    fw_weight[i][:lengths[i]] = 1 - pt.linspace(start=0, end=1, steps=lengths[i]).view(-1, 1).to(device)
    # print("LENGTHS : ", lengths[i])
    # print(pt.linspace(start=0, end=1, steps=lengths[i]+1).view(-1, 1).to(device).shape)
    # print(fw_weight[i][:lengths[i]+1].shape)
    # print(fw_weight[i][:lengths[i]+2].shape)
    # exit()
    # Backward prediction weight
    bw_weight[i][:lengths[i]] = pt.linspace(start=0, end=1, steps=lengths[i]).view(-1, 1).to(device)

  return pt.cat((fw_weight, bw_weight), dim=2)

def print_loss(loss_list, name):
  loss_dict = loss_list[0]
  loss = loss_list[1]
  print('   [##] {}...'.format(name), end=' ')
  print('{} Loss : {:.3f}'.format(name, loss.item()))
  for idx, loss in enumerate(loss_dict.keys()):
    if idx == 0:
      print('   ======> {} : {:.3f}'.format(loss, loss_dict[loss]), end=', ')
    elif idx == len(loss_dict.keys())-1:
      print('{} : {:.3f}'.format(loss, loss_dict[loss]))
    else:
      print('{} : {:.3f}'.format(loss, loss_dict[loss]), end=', ')

def add_flag_noise(flag, lengths):
  flag = flag * 0
  return flag

def select_uv_recon(input_dict, pred_dict, in_f_noisy):
  if args.recon == 'ideal_uv':
    return input_dict['input'][..., [0, 1]]
  elif args.recon == 'noisy_uv' and 'uv' not in args.pipeline:
    return in_f_noisy
  elif 'uv' in args.pipeline :
    if args.recon == 'noisy_uv':
      return in_f_noisy
    elif args.recon =='pred_uv':
      return pred_dict['model_uv']
  else:
    return input_dict['input_dict'][..., [0, 1]]

def load_ckpt_train(model_dict, optimizer, lr_scheduler):
  if args.load_ckpt == 'best':
    load_ckpt = '{}/{}/{}_best.pth'.format(args.save_ckpt + args.wandb_tags.replace('/', '_'), args.wandb_name, args.wandb_name)
  elif args.load_ckpt == 'lastest':
    load_ckpt = '{}/{}/{}_lastest.pth'.format(args.save_ckpt + args.wandb_tags.replace('/', '_'), args.wandb_name, args.wandb_name)
  else:
    print("[#] The load_ckpt should be \'best\' or \'lastest\' keywords...")
    exit()

  if os.path.isfile(load_ckpt):
    print("[#] Found the ckpt ===> {}".format(load_ckpt))
    ckpt = pt.load(load_ckpt, map_location='cuda:0')
    # Load optimizer, learning rate, decay and scheduler parameters
    for model in ckpt['model_cfg'].keys():
      model_dict[model].load_state_dict(ckpt[model])

    optimizer.load_state_dict(ckpt['optimizer'])
    lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
    start_epoch = ckpt['epoch']
    min_val_loss = ckpt['min_val_loss']
    annealing_scheduler = ckpt['annealing_scheduler']
    return model_dict, optimizer, start_epoch, lr_scheduler, min_val_loss, annealing_scheduler

  else:
    print("[#] ckpt not found...")
    exit()

def load_ckpt_predict(model_dict, ckpt):
  print("="*100)
  print("[#] Model Parameters")
  for model in model_dict.keys():
    for k, v in model_dict[model].named_parameters():
      print("===> ", k, v.shape)
  print("="*100)
  if os.path.isfile(ckpt):
    print("[#] Found the ckpt ===> {}".format(ckpt))
    ckpt = pt.load(ckpt, map_location='cuda:0')
    # Load optimizer, learning rate, decay and scheduler parameters
    for model in ckpt['model_cfg'].keys():
      print("Module ===> {}.....".format(model), end='')
      # Lastest version of dict keys
      model_dict[model].load_state_dict(ckpt[model])
      model_dict['{}'.format(model)].load_state_dict(ckpt['{}'.format(model)])
      print("Loaded!!!")

    return model_dict

  else:
    print("[#] ckpt not found...")
    exit()

def duplicate_at_length(seq, lengths):
  rep_mask = pt.zeros(size=seq.shape).to(device)
  lengths = lengths - 1 # zero-th indexing
  # Masking the duplicate at specific lengths (lastest element)
  for idx in range(rep_mask.shape[0]):
    rep_mask[idx, lengths[idx], :] = 1

  # Use the a value before the last one to replace
  replace_seq = pt.unsqueeze(seq[pt.arange(seq.shape[0]), lengths-1], dim=1)
  duplicated = seq.masked_scatter_(rep_mask==1, replace_seq)
  return duplicated

def yaml_to_args(args):
  with open(args.config_yaml) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
  
  args_dict = vars(args)
  #print("[#] Before")
  #print(args)
  for k in config:
    print(k, " ===> ", args_dict[k])
    if args_dict[k] is not None and config[k] is None:
      # User specified the parameters via args
      continue
    else:
      # Load from config file
      args_dict[k] = config[k]

  return args

def show_dataset_info(dataloader, set):

  # Dataset format
  # Trajectory path : (x0, y0) ... (xn, yn)
  print("======================================================{} - Summary Batch (batch_size = {})=========================================================================".format(set, args.batch_size))
  for key, batch in enumerate(dataloader):
    print("Input batch [{}] : batch={}, lengths={}, mask={}".format(key, batch['input'][0].shape, batch['input'][1].shape, batch['input'][2].shape))
    print("gt batch [{}] : batch={}, lengths={}, mask={}".format(key, batch['gt'][0].shape, batch['gt'][1].shape, batch['gt'][2].shape))
    # Test RNN/LSTM Step
    # 1.Pack the padded
    packed = pt.nn.utils.rnn.pack_padded_sequence(batch['input'][0], batch_first=True, lengths=batch['input'][1], enforce_sorted=False)
    # 2.RNN/LSTM model
    # 3.Unpack the packed
    unpacked = pt.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True, padding_value=-1000.0)
    print("Unpacked equality : ", pt.eq(batch['input'][0], unpacked[0]).all())
    print("===============================================================================================================================================================")

def cumsum(seq, t_0=None):
  '''
  Perform a cummulative summation to any seq along the time-dimension
  Input : 
    - seq = input sequence (batch, seq_len, features)
    - t_0 => if this given, we concat this to be the first time step (batch, 1, features)
  '''

  if t_0 is not None:
    # output : perform cumsum along the sequence_length axis
    seq = pt.stack([pt.cat([t_0[i][:, [0]], seq[i]]) for i in range(seq.shape[0])])
  
  seq = pt.cumsum(seq, dim=1)
  return seq

def uv_noise(uv):
    '''
    Add noise to uv tracking
    Input : 
        1. UV-coordinates in shape = (batch, seq_len, 2)
    Output : 
        1. Noisy-UV in shape = (batch, seq_len, 2)
    '''
    # Generate noise
    noise_u = np.random.normal(loc=0.0, scale=5, size=uv[..., [0]].shape)
    noise_v = np.random.normal(loc=0.0, scale=5, size=uv[..., [1]].shape)
    noise_uv = np.concatenate((noise_u, noise_v), axis=-1)

    # Masking noise
    noise_mask_u = np.random.uniform(size=uv[..., [0]].shape) > np.random.random_sample(size=1)
    noise_mask_v = np.random.uniform(size=uv[..., [1]].shape) > np.random.random_sample(size=1)
    noise_mask_uv = np.concatenate((noise_mask_u, noise_mask_v), axis=-1)

    # Noise index selection
    n = round(uv.shape[0] * 0.5)
    noise_idx = np.random.choice(a=uv.shape[0], size=(n,), replace=False)

    # Apply noise
    noisy_uv = uv.clone()
    noisy_uv[noise_idx] = (uv[noise_idx] + noise_uv[noise_idx] * noise_mask_uv[noise_idx]).float()
    noisy_uv = uv + noise_uv * noise_mask_uv

    return noisy_uv


def add_noise(cam_dict, in_f=None):
  '''
  - Adding noise to uv tracking
  - Need to recalculate all features since
    => UV changed -> Ray changed -> Intersect/Azimuth/Elevation changed
  '''

  # UV-noise
  noisy_uv = uv_noise(uv=cam_dict['tracking'])

  #print(noisy_uv[0][:10], cam_dict['tracking'][0][:10])
  #print(noisy_uv[0][:10] - cam_dict['tracking'][0][:10])
  # Cast-Ray
  noisy_ray = utils_transform.cast_ray(uv=noisy_uv, I=cam_dict['I'], E=cam_dict['E'])
  #noisy_ray = utils_transform.cast_ray(uv=cam_dict['tracking'], I=cam_dict['I'], E=cam_dict['E'])

  # Intersect Plane
  noisy_intr = utils_transform.ray_to_plane(E=cam_dict['E'], ray=noisy_ray)
  #print(noisy_intr[0][:10], in_f[0][:10, :3])
  #print(noisy_intr[0][:10] - in_f[0][:10, :3])

  # New azimuth
  noisy_azim = utils_transform.compute_azimuth(ray=noisy_ray)
  #print(noisy_azim[0][:10, :], in_f[0][:10, [4]])
  #print(noisy_azim[0][:10, :] - in_f[0][:10, [4]])

  # New elevation
  noisy_elev = utils_transform.compute_elevation(intr=noisy_intr, E=cam_dict['E'])
  #print(noisy_elev[0][:10, :], in_f[0][:10, [3]])
  #print(noisy_elev[0][:10, :] - in_f[0][:10, [3]])

  # Replace in_f
  noise_in_f = pt.cat((noisy_intr, noisy_elev, noisy_azim), dim=-1)

  return noise_in_f.float(), noisy_ray

def generate_input(cam_dict, in_f=None):
  '''
  - Adding noise to uv tracking
  - Need to recalculate all features since
    => UV changed -> Ray changed -> Intersect/Azimuth/Elevation changed
  '''
  # Cast-Ray
  ray = utils_transform.cast_ray(uv=cam_dict['tracking'], I=cam_dict['I'], E=cam_dict['E'])

  # Intersect Plane
  intr = utils_transform.ray_to_plane(E=cam_dict['E'], ray=ray)
  #print(intr[0][:10], in_f[0][:10, :3])
  #print("DIFF : ", intr[0][:10] - in_f[0][:10, :3])
  #print("MAX : ", pt.max(intr[0] - in_f[0][:, :3], dim=1))
  #print("MEAN : ", pt.mean(intr[0] - in_f[0][:, :3]))
  #input("Prev is intr check")

  # New azimuth
  azim = utils_transform.compute_azimuth(ray=ray)
  #print(azim[0][:10, :], in_f[0][:10, [4]])
  #print("DIFF : ", azim[0][:10] - in_f[0][:10, [4]])
  #print("MAX : ", pt.max(azim[0] - in_f[0][:, [4]], dim=1))
  #print("MEAN : ", pt.mean(azim[0] - in_f[0][:, [4]]))
  #input("Prev is azim check")

  # New elevation
  elev = utils_transform.compute_elevation(intr=intr, E=cam_dict['E'])
  #print(elev[0][:10, :], in_f[0][:10, [3]])
  #print("DIFF : ", elev[0][:10] - in_f[0][:10, [3]])
  #print("MAX : ", pt.max(elev[0] - in_f[0][:, [3]], dim=1))
  #print("MEAN : ", pt.mean(elev[0] - in_f[0][:, [3]]))
  #input("Prev is elev check")

  # Replace in_f
  in_f = np.concatenate((intr, elev, azim), axis=-1)

  # Cast to tensor
  ray = pt.tensor(ray).float().to(device)
  in_f = pt.tensor(in_f).float().to(device)
  return in_f, ray

def save_reconstructed(eval_metrics, trajectory):
  # Take the evaluation metrics and reconstructed trajectory to create the save file for ranking visualization
  lengths = []
  trajectory_all = []
  prediction_all = []
  gt_all = []
  if args.optimize is not None:
    latent_all = []
  flag_all = []
  # Over batch
  for i in range(len(trajectory)):
    # Iterate over each batch
    gt_xyz = trajectory[i][0]
    pred_xyz = trajectory[i][1]
    uv = trajectory[i][2]
    d = trajectory[i][3]
    seq_len = trajectory[i][4]
    if 'eot' in args.pipeline:
      flag_pred = np.concatenate((np.zeros((seq_len.shape[0], 1, 1)), trajectory[i][5]), axis=1)
      flag_gt = trajectory[i][7]
    if args.optimize is not None:
      latent_gt = trajectory[i][8]
      latent_opt = trajectory[i][6]
    for j in range(seq_len.shape[0]):
      # Iterate over each trajectory
      each_trajectory = np.concatenate((gt_xyz[j][:seq_len[j]], pred_xyz[j][:seq_len[j]], uv[j][:seq_len[j]], d[j][:seq_len[j]].reshape(-1, 1)), axis=1)
      lengths.append(seq_len)
      trajectory_all.append(each_trajectory)
      gt_all.append(gt_xyz[j][:seq_len[j]])
      prediction_all.append(pred_xyz[j][:seq_len[j]])
      # Latent columns => [gt, pred]
      if args.optimize is not None:
        latent_all.append(np.concatenate((latent_gt[j][:seq_len[j]], latent_opt[j][:seq_len[j]]), axis=-1))
      if 'eot' in args.pipeline:
        flag_all.append(np.concatenate((flag_gt[j][:seq_len[j]], flag_pred[j][:seq_len[j]]), axis=-1))

  # Save to file
  save_file_suffix = args.load_ckpt.split('/')[-2] + '_{}'.format(args.label)
  save_path = '{}/{}'.format(args.save_cam_traj, save_file_suffix)
  initialize_folder(save_path)
  np.save(file='{}/{}_trajectory'.format(save_path, save_file_suffix), arr=np.array(trajectory_all))
  np.save(file='{}/{}_trajectory_gt'.format(save_path, save_file_suffix), arr=np.array(gt_all))
  np.save(file='{}/{}_trajectory_prediction'.format(save_path, save_file_suffix), arr=np.array(prediction_all))

  if 'eot' in args.pipeline:
    np.save(file='{}/{}_trajectory_flag'.format(save_path, save_file_suffix), arr=np.array(flag_all))
  if args.optimize is not None:
    np.save(file='{}/{}_trajectory_latent'.format(save_path, save_file_suffix), arr=np.array(latent_all))

  print("[#] Saving reconstruction to /{}/{}".format(args.savetofile, save_file_suffix))


def save_cam_traj(eval_metrics, trajectory):
  #wandb_name = args.config_yaml.split('/')[-2]
  #wandb_tag = args.config_yaml.split('/')[-1]
  #print(args.wandb_name, args.wandb_tags)
  #print(args.config_yaml.split('/'))
  save_path = '{}/tags_{}/{}'.format(args.save_cam_traj, args.wandb_tags, args.wandb_name)
  initialize_folder(save_path)
  #print(len(trajectory))
  #print(trajectory[0])
  #print(eval_metrics.keys())
  #print(eval_metrics['RMSE']['loss_3axis'].shape)

  pred = []
  gt = []
  cpos = []
  for i in range(len(trajectory)):
    # Each batch
    gt_tmp =  trajectory[i]['gt']
    pred_tmp =  trajectory[i]['pred']
    seq_len = trajectory[i]['seq_len']
    cpos_tmp = trajectory[i]['cpos']

    for j in range(seq_len.shape[0]):
      # Each trajectory
      gt.append(gt_tmp[j][:seq_len[j]])
      pred.append(pred_tmp[j][:seq_len[j]])
      cpos.append(cpos_tmp[j][:seq_len[j]])

  data = {'gt':gt, 'pred':pred, 'cpos':cpos}
  np.save(file='{}/reconstructed.npy'.format(save_path), arr=data)
  print("[#] Saving reconstruction to {}".format(save_path))

def augment(batch, aug_col=eot):

  len_ = np.array([trajectory.shape[0] for trajectory in batch])

  if args.augment == 'perc':
    # Split by percentage
    perc = [0.25, 0.50, 0.75, 1]
    perc = np.random.choice(a=perc, size=len(batch))
    len_aug = np.ceil(len_.copy() * perc).astype(int)

    for i in range(len(batch)):
      #print("L : ", len_[i], "L_aug : ", len_aug[i], "P : ", perc[i])
      h = len_[i] - len_aug[i] if len_[i] != len_aug[i] else 1
      try :
        start = np.random.randint(low=0, high=h, size=1)[0]
      except ValueError:
        print("TRY FAILED : ", len_[i], len_aug[i])
        exit()
      end = start + len_aug[i]
      batch[i] = batch[i][start:end]
  
  elif args.augment == 'eot':
    # Split by EOT
    for i in range(len(batch)):
      aug_pos = np.where(batch[i][:, eot] == 1)[0]
      start = np.random.randint(low=0, high=aug_pos[0], size=1)[0]
      end = np.random.randint(low=aug_pos[0] + (aug_pos[1] - aug_pos[0])//2, high=aug_pos[1], size=1)[0]
      batch[i] = batch[i][start:end]

  else:
    print("[#] Augmentation method is incorrect!!!")
    exit()

  return batch 
    