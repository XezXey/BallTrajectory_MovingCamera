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
from models.refinement_module import Refinement_Module
from models.flag_module import Flag_Module

# GPU initialization
if pt.cuda.is_available():
  device = pt.device('cuda')
else:
  device = pt.device('cpu')

args=None
""" 
# Not og dataset
features = ['x', 'y', 'z', 'u', 'v', 'd', 'intr_x', 'intr_y', 'intr_z', 'ray_x', 'ray_y', 'ray_z', 
            'eot', 'cd', 'rad', 'f_sin', 'f_cos', 'fx', 'fy', 'fz', 'fx_norm', 'fy_norm', 'fz_norm',
            'intrinsic', 'extrinsic', 'azimuth', 'elevation', 'extrinsic_inv', 'g']
x, y, z, u, v, d, intr_x, intr_y, intr_z, ray_x, ray_y, ray_z, eot, cd, rad, f_sin, f_cos, fx, fy, fz, fx_norm, fy_norm, fz_norm, intrinsic, extrinsic, azimuth, elevation, extrinsic_inv, g = range(len(features))
"""
#"""
features = ['x', 'y', 'z', 'u', 'v', 'd', 'intr_x', 'intr_y', 'intr_z', 'ray_x', 'ray_y', 'ray_z', 
            'eot', 'cd', 'og', 'hw', 'rad', 'f_sin', 'f_cos', 'fx', 'fy', 'fz', 'fx_norm', 'fy_norm', 'fz_norm',
            'intrinsic', 'extrinsic', 'azimuth', 'elevation', 'extrinsic_inv', 'g']
x, y, z, u, v, d, intr_x, intr_y, intr_z, ray_x, ray_y, ray_z, eot, cd, og, hw, rad, f_sin, f_cos, fx, fy, fz, fx_norm, fy_norm, fz_norm, intrinsic, extrinsic, azimuth, elevation, extrinsic_inv, g = range(len(features))
#"""

def share_args(a):
  global args
  args = a

def initialize_folder(path):
  if not os.path.exists(path):
      os.makedirs(path)

def random_seed():
  #print('-' * 50)
  #print("[#] Seeding...")
  #print("OLD RNG : ", pt.get_rng_state())
  pt.manual_seed(time.time())
  #print("NEW RNG : ", pt.get_rng_state())
  #print('-' * 50)

def get_selected_cols(args, pred):
  # Flag/Extra features columns
  features_col = []
  if 'eot' in args.selected_features:
    features_col.append(eot)
  if 'og' in args.selected_features:
    features_col.append(og)
  if 'cd' in args.selected_features:
    features_col.append(cd)
  if 'rad' in args.selected_features:
    features_col.append(rad)
  if 'f_sin' in args.selected_features:
    features_col.append(f_sin)
  if 'f_cos' in args.selected_features:
    features_col.append(f_cos)
  if 'fx' in args.selected_features:
    features_col.append(fx)
  if 'fy' in args.selected_features:
    features_col.append(fy)
  if 'fz' in args.selected_features:
    features_col.append(fz)
  if 'fx_norm' in args.selected_features:
    features_col.append(fx_norm)
  if 'fy_norm' in args.selected_features:
    features_col.append(fy_norm)
  if 'fz_norm' in args.selected_features:
    features_col.append(fz_norm)

  if pred=='height' and args.env=='unity':
    input_col = [intr_x, intr_y, intr_z, elevation, azimuth]
    gt_col = [x, y, z]
    features_col = features_col
  else :
    input_col = "[u, v]"
    gt_col = "[x, y, z] (if existed)"
    features_col = "No features cols from real data"

  print('='*47 + "Features" + '='*47)
  print('Prediction = {}, Environment = {}'.format(pred, args.env))
  print("Available features : ", ['{}-{}'.format(features[idx], idx) for idx in range(len(features))])
  print("Selected features : ", features_col)
  print("1. input_col = ", input_col)
  print("2. gt_col = ", gt_col)
  print('='*100)
  return input_col, gt_col, features_col

def get_extra_fsize(module):

    #############################################
    ############ EXTRA INPUT FEATURES ###########
    #############################################
    extra_in = sum(module['extra_in'])                  # Extra input to module (e.g. eot)
    latent_in = sum(module['latent_in'] )               # Extra latent(aux) input to module (e.g. force, acc)

    #############################################
    ########### EXTRA OUTPUT FEATURES ###########
    #############################################
    extra_out = sum(module['extra_out'])                # Extra output to module (e.g. force, acc)
    latent_out = sum(module['latent_out'])             # Extra latent(aux) from module (e.g. force, acc)

    return extra_in, extra_out, latent_in, latent_out
    
def get_model(args):

  #############################################
  ############## Model I/O Size ###############
  #############################################

  model_dict = {}
  model_cfg = {}
  for module_name in args.pipeline:
    model = None
    module = args.pipeline[module_name]                                         # Module
    extra_in, extra_out, latent_in, latent_out = get_extra_fsize(module)        # Extra input/output features
    in_node = module['in_node'] + extra_in + latent_in                          # Input node
    out_node = module['out_node'] + extra_out + latent_out                      # Output node
    arch = module['arch']                                                       # Architecture

    if module_name == 'height': 
      model = Height_Module(in_node=in_node, out_node=out_node, 
                    batch_size=args.batch_size, trainable_init=module['trainable_init'], 
                    is_bidirectional=module['bidirectional'], 
                    mlp_hidden=module['mlp_hidden'], mlp_stack=module['mlp_stack'],
                    rnn_hidden=module['rnn_hidden'], rnn_stack=module['rnn_stack'],
                    attn=module['attn'])

      model_cfg[module_name] = {'in_node':in_node, 'out_node':out_node,
                      'mlp_hidden':module['mlp_hidden'], 'mlp_stack':module['mlp_stack'],
                      'rnn_hidden':module['rnn_hidden'], 'rnn_stack':module['rnn_stack'],
                      'attn':module['attn']}

    elif module_name == 'refinement': 
      model = Refinement_Module(in_node=in_node, out_node=out_node, 
                    batch_size=args.batch_size, trainable_init=module['trainable_init'], 
                    is_bidirectional=module['bidirectional'], 
                    mlp_hidden=module['mlp_hidden'], mlp_stack=module['mlp_stack'],
                    rnn_hidden=module['rnn_hidden'], rnn_stack=module['rnn_stack'],
                    attn=module['attn'])

      model_cfg[module_name] = {'in_node':in_node, 'out_node':out_node,
                      'mlp_hidden':module['mlp_hidden'], 'mlp_stack':module['mlp_stack'],
                      'rnn_hidden':module['rnn_hidden'], 'rnn_stack':module['rnn_stack'],
                      'attn':module['attn']}

    elif module_name == 'flag': 
      model = Flag_Module(in_node=in_node, out_node=out_node, 
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
    #for model in ckpt['model_cfg'].keys():
    for model in model_dict.keys():
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
  exception = ['load_ckpt', 'wandb_mode', 'dataset_test_path', 'save_cam_traj', 'optim_init_h', 'optim_latent', 'wandb_resume', 'save_suffix']
  for k in args_dict.keys():
    if k in exception:
      continue
    if args_dict[k] is not None:
      # User specified the parameters via args
      continue
    elif config[k] is not None :
      # Load from config file
      args_dict[k] = config[k]
    else:
      print("[#] Config \"{}\" is missing...!".format(k))
      exit()
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

def pad_at_length(tensor, lengths):
  '''
  Pad 0 at a specific length : this used to handle dt-computation since it's [1:] - [:-1] and last timestep should not be the (0 - t_n)
  Input : 
    1. tensor : tensor to be padded in shape (batch_size, seq_len, 5)
    2. lengths : list of seq_len
  '''
  pad_0 = pt.zeros(size=(tensor.shape[0], 1, tensor.shape[2])).to(device)
  tensor_pad = pt.cat((tensor, pad_0), dim=1)
  for i in range(lengths.shape[0]):
    cutoff = lengths[i]
    s = tensor[i, :cutoff, :]
    e = tensor[i, cutoff:, :]
    tmp = pt.cat((s, pad_0[i], e), dim=0)
    tensor_pad[i] = tmp

  return tensor_pad

def cumsum(seq, t_0=None):
  '''
  Perform a cummulative summation to any seq along the time-dimension
  Input : 
    - seq = input sequence (batch, seq_len, features)
    - t_0 => if this given, we concat this to be the first time step (batch, 1, features)
  '''

  if t_0 is not None:
    # output : perform cumsum along the sequence_length axis
    #seq = pt.stack([pt.cat([t_0[i][:, [0]], seq[i]]) for i in range(seq.shape[0])])
    seq = pt.cat((t_0, seq), dim=1)
  
  seq = pt.cumsum(seq, dim=1)
  return seq

def refinement_noise(h, xyz, cam_dict, recon_dict, canon_dict, input_dict, set_):
  if args.noise and set_ == 'train':
    noise_sd = args.pipeline['refinement']['noise_sd']
    if args.pipeline['refinement']['noise_space'] == 'h':
      # Noise on height
      noise_h = pt.normal(mean=0.0, std=noise_sd, size=h.shape).to(device)
      height = h + noise_h
      xyz = utils_transform.reconstruct(height, cam_dict, recon_dict, canon_dict)
    elif args.pipeline['refinement']['noise_space'] == 'dh':
      dh = h[:, 1:, :] - h[:, :-1, :]
      noise_dh = pt.normal(mean=0.0, std=noise_sd, size=dh.shape).to(device)
      height = cumsum(t_0=h[:, [0], :], seq=dh + noise_dh)
      xyz = utils_transform.reconstruct(height, cam_dict, recon_dict, canon_dict)
    elif args.pipeline['refinement']['noise_space'] == 'xyz':
      # 3D augmentation
      raise NotImplemented
    elif args.pipeline['refinement']['noise_space'] == 'dxyz':
      # 3D augmentation
      raise NotImplemented
    elif args.pipeline['refinement']['noise_space'] == 'const_h':
      start = time.time()
      L = pt.unsqueeze(input_dict['lengths'], dim=-1).to(device)
      n_mask = 3  # 3 blocks per sequence
      mask_rl = pt.rand(n_mask).to(device)
      mask_ratio = pt.unsqueeze(mask_rl/pt.sum(mask_rl), dim=0).to(device)
      mask_L = (mask_ratio * L-1).int()
      const_h = h
      for i in range(h.shape[0]):
        m = mask_L[i]
        L_list = np.cumsum([1] + m.cpu().numpy().tolist())
        for j in range(len(L_list)-1):
          s = L_list[j]
          e = L_list[j+1]
          if np.random.rand(1) > 0.5:
            const_h[i][s:e] = pt.mean(h[i][s:e])
          else:
            const_h[i][s:e] = h[i][s:e]
      height = const_h
      xyz = utils_transform.reconstruct(height, cam_dict, recon_dict, canon_dict)
    elif args.pipeline['refinement']['noise_space'] == None:
      # Retain h, xyz
      height = h
      xyz = xyz
    else:
      raise ValueError("[#] Refinement noise is invalid.") 
  else:
    # Retain h, xyz
    height = h
    xyz = xyz

  return xyz, height

def uv_noise(uv):
    '''
    Add noise to uv tracking
    Input : 
        1. UV-coordinates in shape = (batch, seq_len, 2)
    Output : 
        1. Noisy-UV in shape = (batch, seq_len, 2)
    '''
    # Generate noise
    noise_sd = args.pipeline['height']['noise_sd']
    noise_u = pt.normal(mean=0.0, std=noise_sd, size=uv[..., [0]].shape)
    noise_v = pt.normal(mean=0.0, std=noise_sd, size=uv[..., [1]].shape)
    noise_uv = pt.cat((noise_u, noise_v), axis=-1).to(device)

    # Masking noise
    noise_mask_u = pt.rand(size=uv[..., [0]].shape) > pt.rand(1)
    noise_mask_v = pt.rand(size=uv[..., [1]].shape) > pt.rand(1)
    noise_mask_uv = pt.cat((noise_mask_u, noise_mask_v), axis=-1).to(device)

    # Noise index selection
    n = round(uv.shape[0] * 0.5)
    noise_idx = np.random.choice(a=uv.shape[0], size=(n,), replace=False)

    # Apply noise
    noisy_uv = uv.clone()
    noisy_uv[noise_idx] = (uv[noise_idx] + noise_uv[noise_idx] * noise_mask_uv[noise_idx]).float()
    noisy_uv = uv + noise_uv * noise_mask_uv

    return noisy_uv

def add_noise(cam_dict):
  '''
  - Adding noise to uv tracking
  - Need to recalculate all features since
    => UV changed -> Ray changed -> Intersect/Azimuth/Elevation changed
  '''
  # UV-noise
  cpos = cam_dict['Einv'][..., 0:3, -1]
  noisy_uv = uv_noise(uv=cam_dict['tracking'])

  # Cast-Ray
  noisy_ray = utils_transform.cast_ray(uv=noisy_uv, I=cam_dict['I'], E=cam_dict['E'], cpos=cpos)
  #noisy_ray = utils_transform.cast_ray(uv=cam_dict['tracking'], I=cam_dict['I'], E=cam_dict['E'])

  # Intersect Horizontal Plane
  noisy_intr_hori = utils_transform.ray_to_plane(ray=noisy_ray, cpos=cpos, plane='horizontal')

  # Intersect Vertical Plane
  noisy_intr_vert = utils_transform.ray_to_plane(ray=noisy_ray, cpos=cpos, plane='vertical')

  # New azimuth
  noisy_azim = utils_transform.compute_azimuth(ray=noisy_ray)

  # New elevation
  noisy_elev = utils_transform.compute_elevation(intr=noisy_intr_hori, cpos=cpos)

  # Replace in_f
  # Replace in_f followed the input variation
  if args.input_variation == 'intr_azim_elev':
    noisy_in_f = pt.cat((noisy_intr_hori, noisy_elev, noisy_azim), axis=-1)
  elif args.input_variation == 'intr_hori_vert':
    noisy_in_f = pt.cat((noisy_intr_hori, noisy_intr_vert), axis=-1)
  else:
    raise ValueError("Input variation is wrong.")

  return noisy_in_f, noisy_ray

def generate_input(cam_dict):
  '''
  - Adding noise to uv tracking
  - Need to recalculate all features since
    => UV changed -> Ray changed -> Intersect/Azimuth/Elevation changed
  '''
  # Cast-Ray
  cpos = cam_dict['Einv'][..., 0:3, -1]
  ray = utils_transform.cast_ray(uv=cam_dict['tracking'], I=cam_dict['I'], E=cam_dict['E'], cpos=cpos)

  # Intersect Horizontal Plane
  intr_hori = utils_transform.ray_to_plane(ray=ray, cpos=cpos, plane='horizontal')

  # Intersect Vertical Plane
  intr_vert = utils_transform.ray_to_plane(ray=ray, cpos=cpos, plane='vertical')

  # New azimuth
  azim = utils_transform.compute_azimuth(ray=ray)

  # New elevation
  elev = utils_transform.compute_elevation(intr=intr_hori, cpos=cpos)

  # Replace in_f followed the input variation
  if args.input_variation == 'intr_azim_elev':
    in_f = pt.cat((intr_hori, elev, azim), axis=-1)
  elif args.input_variation == 'intr_hori_vert':
    in_f = pt.cat((intr_hori, intr_vert), axis=-1)
  else:
    raise ValueError("Input variation is wrong.")

  return in_f, ray

def save_cam_traj(trajectory, cam_dict, n):
  save_path = '{}/tags_{}/{}'.format(args.save_cam_traj, args.wandb_tags, args.wandb_name)
  initialize_folder(save_path)

  pred = []
  gt = []
  cpos = []
  traj_json = {}

  for i in range(len(trajectory)):
    # Each batch
    gt_tmp =  trajectory[i]['gt']
    pred_tmp =  trajectory[i]['xyz']
    pred_refined_tmp =  trajectory[i]['xyz_refined']
    seq_len = trajectory[i]['seq_len']
    cpos_tmp = trajectory[i]['cpos']
    E_tmp = cam_dict['E'].cpu().numpy()
    I_tmp = cam_dict['I'].cpu().numpy()
    uv_tmp = cam_dict['tracking'].cpu().numpy()

    for j in range(seq_len.shape[0]):
      # Each trajectory
      pred.append(pred_tmp[j][:seq_len[j]])
      cpos.append(cpos_tmp[j][:seq_len[j]])

      if (gt is None) or (args.env == 'tennis') and ('refinement' in args.pipeline):
        # Tennis
        json_dat = {"gt" : pred_refined_tmp[j][:seq_len[j]].tolist(),
                    "pred_unrefined" : pred_tmp[j][:seq_len[j]].tolist(),
                    "pred" : pred_refined_tmp[j][:seq_len[j]].tolist(),
                    "uv" : uv_tmp[j][:seq_len[j]].tolist(),
                    "E" : E_tmp[j][:seq_len[j]].tolist(),
                    "I" : I_tmp[j][:seq_len[j]].tolist(),
        }
      else:
        # Unity, Mocap, IPL
        json_dat = {"gt" : gt_tmp[j][:seq_len[j]].tolist(),
                    "pred_unrefined" : pred_tmp[j][:seq_len[j]].tolist(),
                    "pred" : pred_refined_tmp[j][:seq_len[j]].tolist(),
                    "uv" : uv_tmp[j][:seq_len[j]].tolist(),
                    "E" : E_tmp[j][:seq_len[j]].tolist(),
                    "I" : I_tmp[j][:seq_len[j]].tolist(),
        }

      traj_json[j] = json_dat

    if args.save_suffix != '':
      args.save_suffix = '_' + args.save_suffix
    with open("{}/{}{}.json".format(save_path, args.wandb_name, args.save_suffix), "w") as file:
      txt = "var data = " + str(traj_json)
      file.write(txt)

  print("[#] Saving reconstruction to {}".format(save_path))

def mask_from_lengths(lengths, n_rmv, n_dim=1, retain_L=False):
  '''
  Create a mask from list of seq_len
  Input : 
    1. lengths : contains length of each seq in shape(512)
    2. n_rmv : number of points to remove from lengths (helps in creating a dt mask)
    3. n_dim : expands the final dims to (batch_size, seq_len, 1*n_dim)
    4. retain_L : keep the same seq_len by append False at the end
  Output : 
    1. mask in shape (batch_size, seq_len, n_dim)
      - seq_len = lengths - n_rmv  <===> retain_L = False
      - seq_len = lengths          <===> retain_L = True
  '''

  lengths = lengths - n_rmv
  mask_ = pt.arange(pt.max(lengths))[None, None, :].cpu() < lengths[:, None, None].cpu()
  mask_ = pt.transpose(mask_, 1, 2) # Reshape to (batch_size, seq_len, 1)
  mask_ = pt.cat([mask_]*n_dim, dim=2).to(device)

  if retain_L:
    f_pad = pt.tensor([[[False] * n_dim]]*lengths.shape[0]).to(device)
    mask_ = pt.cat((mask_, f_pad), dim=1)

  return mask_

def augment(batch):
  len_ = np.array([trajectory.shape[0] for trajectory in batch])

  # Split by percentage
  perc = 25
  perc = np.random.randint(low=perc, high=100, size=len(batch))[0]/100
  len_aug = np.ceil(len_.copy() * perc).astype(int)

  for i in range(len(batch)):
    h = len_[i] - len_aug[i] if len_[i] != len_aug[i] else 1
    try :
      start = np.random.randint(low=0, high=h, size=1)[0]
    except ValueError:
      print("AUGMENT LENGTH FAILED : ", len_[i], len_aug[i])
      exit()
    end = start + len_aug[i]
    batch[i] = batch[i][start:end]

  return batch 

def aggregation(tensor, lengths, search_h, space='h'):
  # Aggregate the dt output with ramp_weight
  w_ramp = construct_w_ramp(weight_template=pt.zeros(size=(tensor.shape[0], tensor.shape[1]+1, 1)), lengths=lengths)
  f_dim = 1 if space == 'h' else 3

  if search_h is None:
    first_h = pt.zeros(size=(tensor.shape[0], 1, f_dim)).to(device)
    last_h = pt.zeros(size=(tensor.shape[0], 1, f_dim)).to(device)
  else:
    first_h = search_h['first_h']
    last_h = search_h['last_h']

  # forward aggregate
  h_fw = cumsum(seq=tensor, t_0=first_h)
  # backward aggregate
  pred_h_bw = reverse_masked_seq(seq=-tensor, lengths=lengths-1) # This fn required len(seq) of dt-space
  h_bw = cumsum(seq=pred_h_bw, t_0=last_h)
  h_bw = reverse_masked_seq(seq=h_bw, lengths=lengths) # This fn required len(seq) of t-space(after cumsum)
  height = pt.sum(pt.cat((h_fw, h_bw), dim=2) * w_ramp, dim=2, keepdims=True)

  return height