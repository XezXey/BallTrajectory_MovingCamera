from re import search
import torch as pt
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import tqdm
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

def uv_to_input_features(cam_dict, set_):
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

  if args.noise and set_ == 'train':
    in_f, ray = in_f_noisy, ray_noisy
  else:
    in_f, ray = in_f_raw, ray_raw

  # Recon dict for a reconstruction stuff
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

def fw_pass(model_dict, input_dict, cam_dict, gt_dict, latent_dict, set_):
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

  in_f, ray, recon_dict = uv_to_input_features(cam_dict, set_)

  # Canonicalize
  if args.canonicalize:
    canon_dict, in_f = canonicalize_features(Einv=cam_dict['Einv'], intr=in_f[..., [0, 1, 2]], ray=ray, in_f=in_f)
  else:
    canon_dict = {'cam_cl' : None, 'R' : None}

  # Augmentation
  if set_ == 'train' or set_ == 'val':
    search_h = {}
    search_h['first_h'] = gt_dict['gt'][:, [0], [1]]
    search_h['last_h'] = pt.stack([gt_dict['gt'][i, [input_dict['lengths'][i]-1], [1]] for i in range(gt_dict['gt'].shape[0])])
    search_h['first_h'] = pt.unsqueeze(search_h['first_h'], dim=-1)
    search_h['last_h'] = pt.unsqueeze(search_h['last_h'], dim=-1)
    h0 = search_h['first_h']
  elif args.optim_init_h:
    search_h = {}
    search_h['first_h'] = latent_dict['init_h']['first_h'].get_params()
    search_h['last_h'] = latent_dict['init_h']['last_h'].get_params()
    h0 = latent_dict['init_h']['first_h'].get_params()
  elif args.env == 'no_gt' or args.env == 'tennis':
    search_h = None
    h0 = pt.zeros(size=(in_f.shape[0], 1, 1)).to(device)
  else: 
    search_h = None
    h0 = gt_dict['gt'][:, [0], [1]]

  in_f, in_f_orig = input_manipulate(in_f=in_f, module='height', input_dict=input_dict, h0=h0)
  
  ######################################
  ################ Flag ################
  ######################################
  if 'flag' in args.pipeline:
    #print("FLAG : ", in_f.shape)
    i_s = args.pipeline['flag']['i_s']
    o_s = args.pipeline['flag']['o_s']
    in_f = add_latent(in_f=in_f, input_dict=input_dict, latent_dict=latent_dict, module='flag')
    pred_flag, _ = model_dict['flag'](in_f=in_f, lengths=input_dict['lengths']-1 if (i_s == 'dt' and o_s == 'dt') else input_dict['lengths'])
    pred_dict['flag'] = pred_flag
    in_f = pt.cat((in_f, pred_flag), dim=-1)

  ######################################
  ############### Height ###############
  ######################################
  if 'height' in args.pipeline:
    #print("HEIGHT : ", in_f.shape)
    i_s = args.pipeline['height']['i_s']
    o_s = args.pipeline['height']['o_s']
    in_f = add_latent(in_f=in_f, input_dict=input_dict, latent_dict=latent_dict, module='height')
    pred_h, _ = model_dict['height'](in_f=in_f, lengths=input_dict['lengths']-1 if (i_s == 'dt' and o_s == 'dt') else input_dict['lengths'])
    pred_dict['h'] = pred_h

  height = output_space(pred_h, lengths=input_dict['lengths'], search_h=search_h, module='height')

  xyz = utils_transform.reconstruct(height, cam_dict, recon_dict, canon_dict)

  ######################################
  ############# Refinement #############
  ######################################
  if 'refinement' in args.pipeline:
    if args.pipeline['refinement']['refine'] == 'xyz':
      xyz_, _ = utils_func.refinement_noise(height, xyz, cam_dict, recon_dict, canon_dict, input_dict, set_)
      xyz_ = add_latent(in_f=xyz_, input_dict=input_dict, latent_dict=latent_dict, module='refinement')
      pred_refoff, _ = model_dict['refinement'](in_f=xyz_, lengths=input_dict['lengths'])
      pred_dict['refine_offset'] = pred_refoff
      xyz_refined = xyz_ + pred_refoff
    elif args.pipeline['refinement']['refine'] == 'dxyz':
      raise NotImplemented
    elif args.pipeline['refinement']['refine'] == 'h':
      xyz_, height_ = utils_func.refinement_noise(height, xyz, cam_dict, recon_dict, canon_dict, input_dict, set_)
      height_ = add_latent(in_f=height_, input_dict=input_dict, latent_dict=latent_dict, module='refinement')
      pred_refoff, _ = model_dict['refinement'](in_f=height_, lengths=input_dict['lengths'])
      pred_dict['refine_offset'] = pred_refoff
      height_refined = height_[..., [0]] + pred_refoff
      xyz_refined = utils_transform.reconstruct(height_refined, cam_dict, recon_dict, canon_dict)
    elif args.pipeline['refinement']['refine'] == 'dh':
      xyz_, height_ = utils_func.refinement_noise(height, xyz, cam_dict, recon_dict, canon_dict, input_dict, set_)
      dh_ = height_[:, 1:, :] - height_[:, :-1, :]
      dh_ = add_latent(in_f=dh_, input_dict=input_dict, latent_dict=latent_dict, module='refinement')
      pred_refoff, _ = model_dict['refinement'](in_f=dh_, lengths=input_dict['lengths']-1)
      pred_dict['refine_offset'] = pred_refoff
      dh_refined = dh_[..., [0]] + pred_refoff
      height_refined = utils_func.aggregation(tensor=dh_refined, lengths=input_dict['lengths'], search_h=search_h)
      xyz_refined = utils_transform.reconstruct(height_refined, cam_dict, recon_dict, canon_dict)
    elif args.pipeline['refinement']['refine'] == 'infref_h':
      xyz_, height_ = utils_func.refinement_noise(height, xyz, cam_dict, recon_dict, canon_dict, input_dict, set_)
      height_ = pt.cat((height_, in_f_orig), dim=2)
      height_ = add_latent(in_f=height_, input_dict=input_dict, latent_dict=latent_dict, module='refinement')
      pred_refoff, _ = model_dict['refinement'](in_f=height_, lengths=input_dict['lengths'])
      pred_dict['refine_offset'] = pred_refoff
      height_refined = height_[..., [0]] + pred_refoff
      xyz_refined = utils_transform.reconstruct(height_refined, cam_dict, recon_dict, canon_dict)
  
  else:
    xyz_refined = None
    xyz_ = None

  # Decoanonicalize
  if args.canonicalize:
    xyz = utils_transform.canonicalize(pts=xyz, R=canon_dict['R'], inv=True)
    if xyz_refined is not None:
      xyz_refined = utils_transform.canonicalize(pts=xyz_refined, R=canon_dict['R'], inv=True)
      xyz_ = utils_transform.canonicalize(pts=xyz_, R=canon_dict['R'], inv=True)

  pred_dict['xyz'] = xyz
  pred_dict['xyz_refnoise'] = xyz_
  pred_dict['xyz_refined'] = xyz_refined

  return pred_dict, in_f

def fw_pass_optim(model_dict, input_dict, cam_dict, gt_dict, latent_dict):
  '''
  Forward Pass with an optimization.
  '''
  
  utils_func.random_seed() # Seeding the initial latent
  # Training mode to peserve the gradient
  train_mode(model_dict=model_dict)
  # Construct latent
  latent_dict = create_latent(latent_dict, input_dict, model_dict)
  t = tqdm.trange(500, leave=True)

  patience = 10
  count = 0
  prev_loss = 2e16
  for _ in t:
    #t0 = {}
    #for name, param in model_dict['refinement'].named_parameters():
    #for name, param in latent_dict['height'].named_parameters():
    #  if param.requires_grad:
    #    t0[name] = param.data.clone()
    pred_dict, in_test = fw_pass(model_dict=model_dict, input_dict=input_dict, cam_dict=cam_dict, gt_dict=gt_dict, latent_dict=latent_dict)
    loss_dict, loss = optimization_loss(input_dict=input_dict, pred_dict=pred_dict, gt_dict=gt_dict, cam_dict=cam_dict, latent_dict=latent_dict)
    for name, optim in latent_dict.items():
      if optim is not None:
        if name == 'init_h':
          optim['first_h'](loss)
          optim['last_h'](loss)
        else:
          optim(loss)
          #print(pt.cat((pred_dict['refine_offset'][0], pred_dict['xyz'][0][..., [1]]), dim=-1))
          #print(pt.cat((pred_dict['refine_offset'][0], pred_dict['xyz'][0][..., [1]], pred_dict['xyz_refined'][0][..., [1]]), dim=-1))
          #print(pt.cat((pred_dict['refine_offset'][0], pred_dict['xyz'][0][..., [1]], pred_dict['xyz_refined'][0][..., [1]], pred_dict['xyz'][0][..., [1]] + pred_dict['refine_offset'][0]), dim=-1)[:input_dict['lengths'][0]+1])
          #print(pt.cat((pred_dict['refine_offset'][0], pred_dict['xyz_refined'][0]), dim=-1))

    txt_loss = ''
    for k, v in loss_dict.items():
      txt_loss += '{}={:.3f}, '.format(k, v)
    t.set_description("Optimizing... (Loss = {:.5f}, {})".format(loss.item(), txt_loss))
    t.refresh()

    # Early Stopping mechanism
    if prev_loss <= loss.item():
      count+=1
    else:
      count=0
    prev_loss = loss.item()
    if count >= patience:
      break
    #t1 = {}
    #for name, param in latent_dict['height'].named_parameters():
    #for name, param in model_dict['refinement'].named_parameters():
    #  if param.requires_grad:
    #    t1[name] = param.data
    #for name in t1.keys():
    #  print(t1[name] - t0[name])

  return pred_dict, in_test

def add_latent(in_f, module, input_dict, latent_dict):
  #print("[#] Module : ", module)
  latent_dim = sum(args.pipeline[module]['latent_in'])
  #print("Latent dim : ", latent_dim)
  #print("in_f shape : ", in_f.shape)
  
  i_s = args.pipeline[module]['i_s']
  o_s = args.pipeline[module]['o_s']

  if latent_dim > 0:
    # Auxialiary features have been used.
    if latent_dict[module] is not None:
      latent = latent_dict[module].get_params()
      in_f = pt.cat((in_f, latent), dim=-1)
    elif 'aux' not in input_dict.keys():
      raise ValueError('[#] Aux features are not provided. Please check --optim_<module>, --env <unity/no_gt>')
    else:
      sel_f = 0 if 'flag' in args.pipeline else 1   # selected_features -1 since first is eot/cd
      latent_idx = 1 - sel_f  # Latent index in selected_features
      aux = input_dict['aux'][..., latent_idx:]
      #print("aux shape : ", aux.shape)
      aux = aux_space(aux, lengths=input_dict['lengths']-1 if (i_s == 'dt' and o_s == 'dt') else input_dict['lengths'], i_s=i_s)
      in_f = pt.cat((in_f, aux), dim=-1)
  else:
    in_f = in_f
  
  return in_f

def create_latent(latent_dict, input_dict, model_dict):
  # Optimizae for initial height
  batch_size = input_dict['input'].shape[0]
  if args.optim_init_h:
    search_h = {}
    optim_first_h = Optimization(shape=(batch_size, 1, 1), name='init_first_h')
    optim_last_h = Optimization(shape=(batch_size, 1, 1), name='init_last_h')
    optim_first_h.train()
    optim_last_h.train()
    search_h['first_h'] = optim_first_h
    search_h['last_h'] = optim_last_h
    latent_dict['init_h'] = search_h

  # Optimize for latent variables
  if args.optim_latent:
    for module in args.pipeline:
      latent_dim = sum(args.pipeline[module]['latent_in'])
      if latent_dim > 0:
        print("[#] Module : {} have latent to be optimized.".format(module))
        seq_len = input_dict['input'].shape[1] - 1 if args.pipeline[module]['i_s'] == 'dt' and args.pipeline[module]['o_s'] == 'dt' else input_dict['input'].shape[1]
        #seq_len = input_dict['input'].shape[1]
        optim_latent = Optimization(shape=(batch_size, seq_len, latent_dim), name=module)
        optim_latent.train()
        latent_dict[module] = optim_latent
  
  return latent_dict
    
def input_manipulate(in_f, module, input_dict, h0=None):
  '''
  Prepare input to have correct space(t, dt, tdt), sin-cos
  Input : 
    1. in_f : input features in shape (batch, seq_len, 5)
  Output : 
    1. in_f : input features after change into specified space(t, dt, tdt) and sin-cos
  '''
  i_s = args.pipeline[module]['i_s']
  o_s = args.pipeline[module]['o_s']

  if args.input_variation == 'intr_azim_elev':
    # Convert degree to radian
    in_f[..., [3]] = in_f[..., [3]] * np.pi / 180
    in_f[..., [4]] = in_f[..., [4]] * np.pi / 180
    if args.sc == 'azim':
      in_f[..., [4]] = in_f[..., [4]] * np.pi / 180
      azim_sc = pt.cat((pt.sin(in_f[..., [4]]), pt.cos(in_f[..., [4]])), axis=2)
      in_f = pt.cat((in_f[..., [0, 1, 2]], in_f[..., [3]], azim_sc), axis=2)
    elif args.sc == 'elev':
      elev_sc = pt.cat((pt.sin(in_f[..., [3]]), pt.cos(in_f[..., [3]])), axis=2)
      in_f = pt.cat((in_f[..., [0, 1, 2]], elev_sc, in_f[..., [4]]), axis=2)
    elif args.sc == 'both':
      azim_sc = pt.cat((pt.sin(in_f[..., [4]]), pt.cos(in_f[..., [4]])), axis=2)
      elev_sc = pt.cat((pt.sin(in_f[..., [3]]), pt.cos(in_f[..., [3]])), axis=2)
      in_f = pt.cat((in_f[..., [0, 1, 2]], elev_sc, azim_sc), axis=2)
    
  print(in_f)
  input()

  in_f_orig = in_f    
  in_f = input_space(in_f, i_s, o_s, lengths=input_dict['lengths'], h0=h0)

  if args.input_variation == 'intr_azim_elev':
    in_f = in_f[..., [0, 2, 3 ,4]]  # Remove intr_y = 0
    in_f_orig = in_f_orig[..., [0, 2, 3 ,4]]  # Remove intr_y = 0
  elif args.input_variation == 'intr_hori_vert':
    in_f = in_f[..., [0, 2, 3, 4]]  # Remove intr_y = 0 and intr_z = 0
    in_f_orig = in_f_orig[..., [0, 2, 3 ,4]]  # Remove intr_y = 0
  else :
    raise ValueError("Input variation is wrong.")

  return in_f, in_f_orig
    
def input_space(in_f, i_s, o_s, lengths, h0):
  '''
  Input : 
    1. in_f : input features(t-space) in shape (batch, seq_len, f_dim) -> e.g. f_dim = (x, y, z, elev, azim)
  Output :
    1. in_f : input features in t/dt/t_dt-space in shape(batch, seq_len, _)
  '''
  if i_s == 'dt':
  # dt
    if o_s == 't':
      dt = in_f[:, 1:, :] - in_f[:, :-1, :]
      dt = pt.cat((in_f[:, [0], :], dt), dim=1)
    elif o_s == 'dt':
      dt = in_f[:, 1:, :] - in_f[:, :-1, :]
    in_f = dt
  elif i_s == 't':
  # t
    in_f = in_f
  elif i_s == 't_dt':
  # t-dt
    dt = in_f[:, 1:, :] - in_f[:, :-1, :]
    dt_pad = utils_func.pad_at_length(tensor=dt, lengths=lengths-1)
    in_f = pt.cat((in_f, dt_pad), dim=2)
  elif i_s == 'h0_dt':
  #h0_dt
    if o_s == 't' or o_s == 't_dt':
      dt = in_f[:, 1:, :] - in_f[:, :-1, :]
      h0_rep = h0.repeat(1, 1, in_f.shape[2])
      in_f = pt.cat((h0_rep, dt), dim=1)
    else:
      raise NotImplemented
  else: 
    raise Exception("I/O space is wrong : {}<=>{}".format(i_s, o_s))

  return in_f

def output_space(pred_h, lengths, module, search_h=None):
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
  i_s = args.pipeline[module]['i_s']
  o_s = args.pipeline[module]['o_s']

  if o_s == 't':
    height = pred_h
  elif o_s == 'dt' or 't_dt':
    if i_s == 'dt':
      dh = pred_h
    elif i_s == 't_dt' or i_s == 't':
      dh = pred_h[:, :-1, :]

    if o_s == 't_dt' and i_s == 't_dt':
      dh = pred_h[:, :-1, [1]]

    # Aggregate the dt output with ramp_weight
    w_ramp = utils_func.construct_w_ramp(weight_template=pt.zeros(size=(dh.shape[0], dh.shape[1]+1, 1)), lengths=lengths)

    if search_h is None:
      first_h = pt.zeros(size=(dh.shape[0], 1, 1)).to(device)
      last_h = pt.zeros(size=(dh.shape[0], 1, 1)).to(device)
    else:
      first_h = search_h['first_h']
      last_h = search_h['last_h']

    # forward aggregate
    h_fw = utils_func.cumsum(seq=dh, t_0=first_h)
    # backward aggregate
    pred_h_bw = utils_func.reverse_masked_seq(seq=-dh, lengths=lengths-1) # This fn required len(seq) of dt-space
    h_bw = utils_func.cumsum(seq=pred_h_bw, t_0=last_h)
    h_bw = utils_func.reverse_masked_seq(seq=h_bw, lengths=lengths) # This fn required len(seq) of t-space(after cumsum)
    height = pt.sum(pt.cat((h_fw, h_bw), dim=2) * w_ramp, dim=2, keepdims=True)

    if o_s == 't_dt':
      height = (pred_h[..., [0]] * 0.5) + (height * 0.5)

  # Hard constraint on Height (y > 0)
  if args.pipeline['height']['constraint_y'] == 'relu':
    relu = pt.nn.ReLU()
    height = relu(height)
  elif args.pipeline['height']['constraint_y'] == 'softplus':
    softplus = pt.nn.Softplus()
    height = softplus(height)
  elif args.pipeline['height']['constraint_y'] == 'lrelu':
    softplus = pt.nn.LeakyReLU(negative_slope=0.01)
    height = softplus(height)
  else:
    pass

  return height

def aux_space_(aux, i_s, lengths):
  if i_s == 'dt':
    tmp = []
    for i in range(aux.shape[0]):
      s = aux[i][0:lengths[i]-1, :]
      e = aux[i][lengths[i]:, :]
      tmp.append(pt.cat((s, e), dim=0))
    aux = pt.stack(tmp)
  elif i_s == 't':
    aux = aux
  else:
    raise NotImplementedError

  return aux

def aux_space(aux, i_s, lengths):
  if i_s == 'dt':
    aux = aux[:, :-1, :]
  elif i_s == 't':
    aux = aux
  else:
    raise NotImplementedError

  return aux

def training_loss(input_dict, gt_dict, pred_dict, cam_dict, anneal_w):
  '''
  Calculate loss
  '''
  #########################################################################
  ############# Trajectory, Below GND, Gravity, Reprojection ##############
  #########################################################################
  traj_loss = utils_loss.TrajectoryLoss(pred=pred_dict['xyz'], gt=gt_dict['gt'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])
  bg_loss = utils_loss.BelowGroundLoss(pred=pred_dict['xyz'], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])
  reprj_loss = utils_loss.ReprojectionLoss(pred=pred_dict['xyz'], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'], cam_dict=cam_dict)

  if 'refinement' in args.pipeline:
    bg_loss_refined = utils_loss.BelowGroundLoss(pred=pred_dict['xyz_refined'], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])
    traj_loss_refined = utils_loss.TrajectoryLoss(pred=pred_dict['xyz_refined'], gt=gt_dict['gt'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])
    reprj_loss_refined = utils_loss.ReprojectionLoss(pred=pred_dict['xyz_refined'], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'], cam_dict=cam_dict)

  ######################################
  ############### Flag #################
  ######################################
  if 'flag' not in pred_dict.keys() or args.env != 'unity':
    flag_loss = pt.tensor(0.).to(device)
  else:
    i_s = args.pipeline['flag']['i_s']
    o_s = args.pipeline['flag']['i_s']
    if i_s == 'dt' and o_s == 'dt':
      pred_dict['flag'] = utils_func.pad_at_length(tensor=pred_dict['flag'], lengths=gt_dict['lengths']-1)
    m = utils_func.mask_from_lengths(lengths=gt_dict['lengths'], n_rmv=1, n_dim=1, retain_L=True)
    flag_loss = utils_loss.EndOfTrajectoryLoss(pred=pred_dict['flag'], gt=input_dict['aux'][..., [0]], mask=m, lengths=gt_dict['lengths'])

  ######################################
  ########### Consine Sim ##############
  ######################################
  #if ('refinement' in args.pipeline):
  #  cosinesim_loss = utils_loss.CosineSimLoss(pred=pred_dict['xyz_refined'], gt=gt_dict['gt'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'], cam_dict=cam_dict, input_dict=input_dict)
  #else:                           
  #  cosinesim_loss = utils_loss.CosineSimLoss(pred=pred_dict['xyz'], gt=gt_dict['gt'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'], cam_dict=cam_dict, input_dict=input_dict)


  ######################################
  ############# Annealing ##############
  ######################################
  if (args.annealing) and ('refinement' in args.pipeline):
    # Annealing
    traj_loss_ = traj_loss + traj_loss_refined * anneal_w
    bg_loss_ = bg_loss + bg_loss_refined * anneal_w
    reprj_loss_ = reprj_loss + reprj_loss_refined * anneal_w
  elif ('refinement' in args.pipeline):
    # No annealing
    traj_loss_ = traj_loss + traj_loss_refined
    bg_loss_ = bg_loss + bg_loss_refined
    reprj_loss_ = reprj_loss + reprj_loss_refined
  else:
    # No refinement
    traj_loss_ = traj_loss
    bg_loss_ = bg_loss
    reprj_loss_ = reprj_loss

  # Combined all losses term
  loss = traj_loss_ + bg_loss_ + flag_loss + reprj_loss_ 
  loss_dict = {"Trajectory Loss":traj_loss_.item(),
               "BelowGnd Loss":bg_loss_.item(),
               "Flag Loss":flag_loss.item(),
               "Reprojection Loss":reprj_loss_.item(),}
               #"ConsineSim Loss":cosinesim_loss.item(),
               #}

  return loss_dict, loss

def optimization_loss(input_dict, pred_dict, cam_dict, gt_dict, latent_dict):
  # Optimization loss term
  ######################################
  ############# Below GND ##############
  ######################################
  below_ground_loss = utils_loss.BelowGroundLoss(pred=pred_dict['xyz'], mask=input_dict['mask'], lengths=input_dict['lengths'])

  ######################################
  ############ Gravity Loss ############
  ######################################
  if ('refinement' in args.pipeline):
    gravity_loss = utils_loss.GravityLoss(pred=pred_dict['xyz_refined'], gt=gt_dict['gt'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])
  else:
    gravity_loss = utils_loss.GravityLoss(pred=pred_dict['xyz'], gt=gt_dict['gt'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])

  ######################################
  ########### Reprojection #############
  ######################################
  if ('refinement' in args.pipeline):
    reprojection_loss = utils_loss.ReprojectionLoss(pred=pred_dict['xyz_refined'], mask=input_dict['mask'][..., [0, 1, 2]], lengths=input_dict['lengths'], cam_dict=cam_dict)
  else:
    reprojection_loss = utils_loss.ReprojectionLoss(pred=pred_dict['xyz'], mask=input_dict['mask'][..., [0, 1, 2]], lengths=input_dict['lengths'], cam_dict=cam_dict)

  ######################################
  ########### Consine Sim ##############
  ######################################
  #if ('refinement' in args.pipeline):
  #  cosinesim_loss = utils_loss.CosineSimLoss(pred=pred_dict['xyz_refined'], gt=gt_dict['gt'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'], cam_dict=cam_dict, input_dict=input_dict, latent_dict=latent_dict)
  #else:                           
  #  cosinesim_loss = utils_loss.CosineSimLoss(pred=pred_dict['xyz'], gt=gt_dict['gt'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'], cam_dict=cam_dict, input_dict=input_dict, latent_dict=latent_dict)

  loss = reprojection_loss + below_ground_loss + gravity_loss #+ cosinesim_loss
  loss_dict = {"BGnd Loss":below_ground_loss.item(), 
               "Grav Loss":gravity_loss.item(), 
               "Reproj Loss":reprojection_loss.item()}
               #"CosSim Loss":cosinesim_loss.item()}

  return loss_dict, loss
