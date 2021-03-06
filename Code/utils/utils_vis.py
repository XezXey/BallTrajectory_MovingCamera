import os, sys, time
from typing import Sequence
sys.path.append(os.path.realpath('../'))
import numpy as np
import torch as pt
import json
import plotly
import math
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
import plotly.graph_objects as go
from itertools import combinations, combinations_with_replacement, product
import wandb
import yaml
import utils.utils_func as utils_func
import utils.transformation as utils_transform
import utils.loss as utils_loss

# marker_dict for contain the marker properties
marker_dict_gt = dict(color='rgba(0, 0, 255, 0.4)', size=4)
marker_dict_noisy = dict(color='rgba(204, 102, 0, 0.4)', size=4)
marker_dict_pred = dict(color='rgba(255, 0, 0, 0.4)', size=4)
marker_dict_refined = dict(color='rgba(153, 0, 77, 0.8)', size=4)
marker_dict_field = dict(color='rgba(0, 255, 0, 0.8)', size=4)
marker_dict_eot = dict(color='rgba(0, 255, 0, 0.4)', size=4)
marker_dict_cam = dict(color='rgba(255, 0, 0, 0.4)', size=10)
marker_dict_intr = dict(color='rgba(255, 127, 14, 1.0)', size=3)
marker_dict_in_refnoisy = dict(color='rgba(0, 255, 255, 1.0)', size=3)

args = None
def share_args(a):
  global args
  args = a

def visualize_layout_update(fig=None, n_vis=5):
  # Save to html file and use wandb to log the html and display (Plotly3D is not working)
  x = 9.5 if args.env == 'tennis' else 10
  y = 0
  z = 17 if args.env == 'tennis' else 10
  for i in range(n_vis*2):
    fig['layout']['scene{}'.format(i+1)].update(xaxis=dict(nticks=5, range=[-x, x]), zaxis=dict(nticks=15, range=[-z, z]), aspectmode='manual', aspectratio=dict(x=1, y=1, z=3))
    #fig['layout']['scene{}'.format(i+1)].update(yaxis = dict(nticks=5, range=[-2, 6],),)
  return fig

def wandb_vis(input_dict_train, gt_dict_train, pred_dict_train, cam_dict_train,
                  input_dict_val, gt_dict_val, pred_dict_val, cam_dict_val, epoch):
    '''
    Make a visualization for logging a prediction to wandb
    Input : 
      1. {input, gt, pred, cam}_dict_train
      2. {input, gt, pred, cam}_dict_val
    '''

    # Visualize by make a subplots of trajectory
    n_vis = 5 if input_dict_train['input'].shape[0] >= 5 else input_dict_train['input'].shape[0]
    # Random the index the be visualize
    train_vis_idx = np.random.choice(np.arange(input_dict_train['input'].shape[0]), size=(n_vis), replace=False)
    val_vis_idx = np.random.choice(np.arange(input_dict_val['input'].shape[0]), size=(n_vis), replace=False)

    ####################################
    ############ Trajectory ############
    ####################################
    fig_traj = make_subplots(rows=n_vis, cols=2, specs=[[{'type':'scatter3d'}, {'type':'scatter3d'}]]*n_vis, horizontal_spacing=0.05, vertical_spacing=0.01)
    visualize_trajectory(pred=pred_dict_train, gt=gt_dict_train['gt'][..., [0, 1, 2]], lengths=gt_dict_train['lengths'], mask=gt_dict_train['mask'][..., [0, 1, 2]], fig=fig_traj, set='Train', vis_idx=train_vis_idx, col=1)
    visualize_trajectory(pred=pred_dict_val, gt=gt_dict_val['gt'][..., [0, 1, 2]], lengths=gt_dict_val['lengths'], mask=gt_dict_val['mask'][..., [0, 1, 2]], fig=fig_traj, set='Validation', vis_idx=val_vis_idx, col=2)
    #visualize_layout_update(fig=fig_traj, n_vis=n_vis)

    ####################################
    ########## Canonicalize ############
    ####################################
    if args.canonicalize:
      fig_canon = make_subplots(rows=n_vis, cols=2, specs=[[{'type':'scatter3d'}, {'type':'scatter3d'}]]*n_vis, horizontal_spacing=0.05, vertical_spacing=0.01)
      visualize_canonicalize(pred=pred_dict_train, gt=gt_dict_train['gt'][..., [0, 1, 2]], lengths=gt_dict_train['lengths'], mask=gt_dict_train['mask'][..., [0, 1, 2]], fig=fig_canon, set='Train', vis_idx=train_vis_idx, col=1, cam_dict=cam_dict_train)
      visualize_canonicalize(pred=pred_dict_val, gt=gt_dict_val['gt'][..., [0, 1, 2]], lengths=gt_dict_val['lengths'], mask=gt_dict_val['mask'][..., [0, 1, 2]], fig=fig_canon, set='Validation', vis_idx=val_vis_idx, col=2, cam_dict=cam_dict_val)
      visualize_layout_update(fig=fig_canon, n_vis=n_vis)

    ####################################
    ############### Ray ################
    ####################################
    visualize_ray(cam_dict=cam_dict_train, input_dict=input_dict_train, fig=fig_traj, set='Train', vis_idx=train_vis_idx, col=1, plane='horizontal', canonicalize=False)
    visualize_ray(cam_dict=cam_dict_val, input_dict=input_dict_val, fig=fig_traj, set='Val', vis_idx=val_vis_idx, col=2, plane='horizontal', canonicalize=False)
    visualize_ray(cam_dict=cam_dict_train, input_dict=input_dict_train, fig=fig_traj, set='Train', vis_idx=train_vis_idx, col=1, plane='vertical', canonicalize=False)
    visualize_ray(cam_dict=cam_dict_val, input_dict=input_dict_val, fig=fig_traj, set='Val', vis_idx=val_vis_idx, col=2, plane='vertical', canonicalize=False)

    if args.canonicalize:
      visualize_ray(cam_dict=cam_dict_train, input_dict=input_dict_train, fig=fig_canon, set='Train', vis_idx=train_vis_idx, col=1, plane='horizontal', canonicalize=True)
      visualize_ray(cam_dict=cam_dict_val, input_dict=input_dict_val, fig=fig_canon, set='Val', vis_idx=val_vis_idx, col=2, plane='horizontal', canonicalize=True)
      visualize_ray(cam_dict=cam_dict_train, input_dict=input_dict_train, fig=fig_canon, set='Train', vis_idx=train_vis_idx, col=1, plane='vertical', canonicalize=True)
      visualize_ray(cam_dict=cam_dict_val, input_dict=input_dict_val, fig=fig_canon, set='Val', vis_idx=val_vis_idx, col=2, plane='vertical', canonicalize=True)
    ####################################
    ############### Flag ###############
    ####################################
    if ('flag' in pred_dict_train.keys()) and ('flag' in pred_dict_val.keys()):
      fig_flag = make_subplots(rows=n_vis, cols=2, specs=[[{'type':'scatter'}, {'type':'scatter'}]]*n_vis, horizontal_spacing=0.05, vertical_spacing=0.01)
      gt_dict_train['flag'] = input_dict_train['aux'][..., [0]]
      gt_dict_val['flag'] = input_dict_val['aux'][..., [0]]
      visualize_flag(pred=pred_dict_train, gt=gt_dict_train, lengths=gt_dict_train['lengths'], mask=gt_dict_train['mask'][..., [0]], fig=fig_flag, set='Train', vis_idx=train_vis_idx, col=1)
      visualize_flag(pred=pred_dict_val, gt=gt_dict_val, lengths=gt_dict_val['lengths'], mask=gt_dict_val['mask'][..., [0]], fig=fig_flag, set='Validation', vis_idx=val_vis_idx, col=2)
      plotly.offline.plot(fig_flag, filename='{}/wandb_vis_flag.html'.format(args.vis_path), auto_open=False)
      wandb.log({'n_epochs':epoch, "Flag(Col1=Train, Col2=Val)":wandb.Html(open('{}/wandb_vis_flag.html'.format(args.vis_path)))})

    fig_traj.update_layout(height=1920, width=1500, autosize=True) # Adjust the layout/axis for pitch scale
    plotly.offline.plot(fig_traj, filename='{}/wandb_vis_traj.html'.format(args.vis_path), auto_open=False)
    if args.canonicalize:
      fig_canon.update_layout(height=1920, width=1500, autosize=True) # Adjust the layout/axis for pitch scale
      plotly.offline.plot(fig_canon, filename='{}/wandb_vis_traj_canon.html'.format(args.vis_path), auto_open=False)

    try:
      wandb.log({'n_epochs':epoch, "Trajectory(Col1=Train, Col2=Val)":wandb.Html(open('{}/wandb_vis_traj.html'.format(args.vis_path)))})
    except ValueError:
      print("[#] Wandb is not init")

def inference_vis(input_dict, gt_dict, pred_dict, cam_dict, latent_dict):
    n_vis = 10 if input_dict['input'].shape[0] > 10 else input_dict['input'].shape[0]
    vis_idx = np.random.choice(np.arange(input_dict['input'].shape[0]), size=(n_vis), replace=False)
    ####################################
    ############ Trajectory ############
    ####################################
    # Variables
    len_ = input_dict['lengths']
    mask = input_dict['mask'][..., [0, 1, 2]]
    if args.env == 'unity' or args.env == 'mocap':
      gt = gt_dict['gt'][..., [0, 1, 2]]
    else:
      gt = None
    fig_traj = make_subplots(rows=math.ceil(n_vis/2), cols=2, specs=[[{'type':'scatter3d'}, {'type':'scatter3d'}]]*math.ceil(n_vis/2), horizontal_spacing=0.05, vertical_spacing=0.01)
    visualize_trajectory(pred=pred_dict, gt=gt, lengths=len_, mask=mask, fig=fig_traj, set='Test', vis_idx=vis_idx[:n_vis//2], col=1)
    visualize_trajectory(pred=pred_dict, gt=gt, lengths=len_, mask=mask, fig=fig_traj, set='Test', vis_idx=vis_idx[n_vis//2:], col=2)
    ####################################
    ############### Ray ################
    ####################################
    visualize_ray(cam_dict=cam_dict, input_dict=input_dict, fig=fig_traj, set='Test', vis_idx=vis_idx[:n_vis//2], col=1)
    visualize_ray(cam_dict=cam_dict, input_dict=input_dict, fig=fig_traj, set='Test', vis_idx=vis_idx[n_vis//2:], col=2)
    # Update layout (if needed)
    fig_traj.update_layout(height=1920, width=1500, autosize=True) # Adjust the layout/axis for pitch scale
    #visualize_layout_update(fig=fig_traj, n_vis=n_vis)
    plotly.offline.plot(fig_traj, filename='{}/pred_vis_traj.html'.format(args.vis_path), auto_open=False)
    ####################################
    ############### Flag ###############
    ####################################
    if ('flag' in pred_dict.keys()):
      # Variables
      len_ = input_dict['lengths']
      if args.env == 'unity':
        gt_dict['flag'] = input_dict['aux'][..., [0]]
      else:
        gt_dict['flag'] = None
      fig_flag = make_subplots(rows=math.ceil(n_vis/2), cols=2, specs=[[{'type':'scatter'}, {'type':'scatter'}]]*math.ceil(n_vis/2), horizontal_spacing=0.05, vertical_spacing=0.01)
      visualize_flag(pred=pred_dict, gt=gt_dict, lengths=len_, mask=mask, fig=fig_flag, set='Test', vis_idx=vis_idx[:n_vis//2], col=1)
      visualize_flag(pred=pred_dict, gt=gt_dict, lengths=len_, mask=mask, fig=fig_flag, set='Test', vis_idx=vis_idx[n_vis//2:], col=2)
      plotly.offline.plot(fig_flag, filename='{}/pred_vis_flag.html'.format(args.vis_path), auto_open=False)

def visualize_ray(cam_dict, input_dict, fig, set, vis_idx, col, plane='horizontal', canonicalize=False):
  # Iterate to plot each trajectory
  cpos = cam_dict['Einv'][..., 0:3, -1]
  ray = utils_transform.cast_ray(uv=cam_dict['tracking'], I=cam_dict['I'], E=cam_dict['E'], cpos=cpos)
  intr = utils_transform.ray_to_plane(ray=ray, cpos=cpos, plane=plane)

  if canonicalize:
    # Canonicalize
    cpos = utils_transform.canonicalize(pts=cpos, R=cam_dict['R'])
    ray = utils_transform.canonicalize(pts=ray[..., [0, 1, 2]], R=cam_dict['R'])
    intr = utils_transform.ray_to_plane(ray=ray, cpos=cpos, plane=plane)
  else:
    ray = ray
    intr = intr

  cpos = cpos.cpu().numpy()
  ray = ray.cpu().numpy()
  intr = intr.cpu().numpy()

  len_ = input_dict['lengths']

  for idx, i in enumerate(vis_idx):
    draw_ray_x = list()
    draw_ray_y = list()
    draw_ray_z = list()
    len_temp = len_[i]
    c_temp = cpos[i][:len_temp]
    r_temp = ray[i][:len_temp]
    intr_temp = intr[i][:len_temp]

    for j in range(c_temp.shape[0]):
        # Cam
        draw_ray_x.append(-c_temp[j, 0])
        draw_ray_y.append(c_temp[j, 1])
        draw_ray_z.append(c_temp[j, 2])
        # Ray
        draw_ray_x.append(-intr_temp[j, 0])
        draw_ray_y.append(intr_temp[j, 1])
        draw_ray_z.append(intr_temp[j, 2])
        # No-interconnect
        draw_ray_x.append(None)
        draw_ray_y.append(None)
        draw_ray_z.append(None)

    ## set the mode to lines to plot only the lines and not the balls/markers
    fig.add_trace(go.Scatter3d(
        x=draw_ray_x,
        y=draw_ray_y,
        z=draw_ray_z,
        mode='lines+markers',
        marker=marker_dict_intr,
        line = dict(width = 0.7, color = 'rgba(102, 0, 204, 0.5)'),
        name='{}-Ray-Trajectory [{}]'.format(set, i).format(i), 
        legendgroup=int(i)
    ), row=idx+1, col=col)
  
    c_temp = np.unique(c_temp, axis=0)
    fig.add_trace(go.Scatter3d(
        x=-c_temp[..., 0],
        y=c_temp[..., 1],
        z=c_temp[..., 2],
        mode='markers',
        marker=marker_dict_cam,
        name='{}-Cam-Trajectory [{}]'.format(set, i).format(i), 
        legendgroup=int(i)
    ), row=idx+1, col=col)

  return fig

def visualize_canonicalize(pred, gt, lengths, mask, vis_idx, set, col, cam_dict, fig=None):
  xyz = pred['xyz'][..., [0, 1, 2]]
  if 'refinement' in args.pipeline:
    xyz_refined = pred['xyz_refined'][..., [0, 1, 2]]
    xyz_refnoise = pred['xyz_refnoise'][..., [0, 1, 2]]
    xyz_refined = utils_transform.canonicalize(pts=xyz_refined, R=cam_dict['R']).cpu().detach().numpy()
    xyz_refnoise = utils_transform.canonicalize(pts=xyz_refnoise, R=cam_dict['R']).cpu().detach().numpy()

  gt = utils_transform.canonicalize(pts=gt, R=cam_dict['R'])
  xyz = utils_transform.canonicalize(pts=xyz, R=cam_dict['R']).cpu().detach().numpy()

  x = 9.5 if args.env == 'tennis' else 5
  y = 0
  z = 17 if args.env == 'tennis' else 5
  field = np.array([[-x, 0, z], [x, 0, z], [x, 0, -z], [-x, 0, -z], [-x, 0, z]]) 

  if gt is not None:
    gt = gt.cpu().detach().numpy()
  # Iterate to plot each trajectory
  for idx, i in enumerate(vis_idx):
    if gt is not None:
      # No-gt
      mse = utils_loss.TrajectoryLoss(pt.tensor(xyz_refined[i]).cuda() if 'refinement' in args.pipeline else pt.tensor(xyz[i]).cuda(), pt.tensor(gt[i]).cuda(), mask=mask[i])
      fig.add_trace(go.Scatter3d(x=-gt[i][:lengths[i], 0], y=gt[i][:lengths[i], 1], z=gt[i][:lengths[i], 2], mode='markers+lines', 
                                marker=marker_dict_gt, name="{}-Ground Truth Trajectory [{}], MSE={:3f}".format(set, i, mse)), row=idx+1, col=col)
    if 'refinement' in args.pipeline:
      # Refinement
      fig.add_trace(go.Scatter3d(x=-xyz_refined[i][:lengths[i], 0], y=xyz_refined[i][:lengths[i], 1], z=xyz_refined[i][:lengths[i], 2], mode='markers+lines', 
                                marker=marker_dict_refined, name="{}-Refined Trajectory [{}]".format(set, i)), row=idx+1, col=col)
      fig.add_trace(go.Scatter3d(x=-xyz_refnoise[i][:lengths[i], 0], y=xyz_refnoise[i][:lengths[i], 1], z=xyz_refnoise[i][:lengths[i], 2], mode='markers+lines', 
                                marker=marker_dict_in_refnoisy, name="{}-In-noisy ref Trajectory[{}]".format(set, i)), row=idx+1, col=col)

    # Reconstructed (from height)
    fig.add_trace(go.Scatter3d(x=-xyz[i][:lengths[i], 0], y=xyz[i][:lengths[i], 1], z=xyz[i][:lengths[i], 2], mode='markers+lines', 
                              marker=marker_dict_pred, name="{}-Estimated Trajectory [{}]".format(set, i)), row=idx+1, col=col)
  
    fig.add_trace(go.Scatter3d(x=field[:, 0], y=field[:, 1], z=field[:, 2], mode='markers+lines', marker=marker_dict_field, name='Field'), row=idx+1, col=col)

def visualize_trajectory(pred, gt, lengths, mask, vis_idx, set, col, fig=None):
  xyz = pred['xyz'][..., [0, 1, 2]].cpu().detach().numpy()
  if 'refinement' in args.pipeline:
    xyz_refined = pred['xyz_refined'][..., [0, 1, 2]].cpu().detach().numpy()
    xyz_refnoise = pred['xyz_refnoise'][..., [0, 1, 2]].cpu().detach().numpy()

  x = 9.5 if args.env == 'tennis' else 5
  y = 0
  z = 17 if args.env == 'tennis' else 5
  field = np.array([[-x, 0, z], [x, 0, z], [x, 0, -z], [-x, 0, -z], [-x, 0, z]]) 

  if gt is not None:
    gt = gt.cpu().detach().numpy()
  # Iterate to plot each trajectory
  for idx, i in enumerate(vis_idx):
    if gt is not None:
      # No-gt
      mse = utils_loss.TrajectoryLoss(pt.tensor(xyz_refined[i]).cuda() if 'refinement' in args.pipeline else pt.tensor(xyz[i]).cuda(), pt.tensor(gt[i]).cuda(), mask=mask[i])
      fig.add_trace(go.Scatter3d(x=-gt[i][:lengths[i], 0], y=gt[i][:lengths[i], 1], z=gt[i][:lengths[i], 2], mode='markers+lines', 
                                marker=marker_dict_gt, name="{}-Ground Truth Trajectory [{}], MSE={:3f}".format(set, i, mse)), row=idx+1, col=col)
    if 'refinement' in args.pipeline:
      # Refinement
      fig.add_trace(go.Scatter3d(x=-xyz_refined[i][:lengths[i], 0], y=xyz_refined[i][:lengths[i], 1], z=xyz_refined[i][:lengths[i], 2], mode='markers+lines', 
                                marker=marker_dict_refined, name="{}-Refined Trajectory [{}]".format(set, i)), row=idx+1, col=col)
      fig.add_trace(go.Scatter3d(x=-xyz_refnoise[i][:lengths[i], 0], y=xyz_refnoise[i][:lengths[i], 1], z=xyz_refnoise[i][:lengths[i], 2], mode='markers+lines', 
                                marker=marker_dict_in_refnoisy, name="{}-In-noisy ref Trajectory[{}]".format(set, i)), row=idx+1, col=col)

    # Reconstructed (from height)
    fig.add_trace(go.Scatter3d(x=-xyz[i][:lengths[i], 0], y=xyz[i][:lengths[i], 1], z=xyz[i][:lengths[i], 2], mode='markers+lines', 
                              marker=marker_dict_pred, name="{}-Estimated Trajectory [{}]".format(set, i)), row=idx+1, col=col)
  
    fig.add_trace(go.Scatter3d(x=field[:, 0], y=field[:, 1], z=field[:, 2], mode='markers+lines', marker=marker_dict_field, name='Field'), row=idx+1, col=col)

def visualize_flag(pred, gt, lengths, mask, vis_idx, set, col, fig=None):
  # Here we use output mask so we need to append the startpos to the pred before multiplied with mask(already included the startpos)
  lengths = lengths.cpu().detach().numpy()
  # detach() for visualization
  flag = pred['flag'][..., [0]].cpu().detach().numpy()
  y = pred['xyz'][..., [1]].cpu().detach().numpy()
  if 'refinement' in args.pipeline:
    y_refined = pred['xyz_refined'][..., [1]].cpu().detach().numpy()
  if gt['flag'] is not None:
    gt_flag = gt['flag'][..., [0]].cpu().detach().numpy()
    gt_y = gt['gt'][..., [1]].cpu().detach().numpy()
  # Iterate to plot each trajectory
  for idx, i in enumerate(vis_idx):
    l = lengths[i] - 1 if args.pipeline['flag']['i_s'] == 'dt' and args.pipeline['flag']['o_s'] == 'dt' else lengths[i]
    if gt['flag'] is not None:
      fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=gt_flag[i][:l].reshape(-1,), mode='markers+lines', marker=marker_dict_gt, name="{}-Ground Truth EOT [{}]".format(set, i)), row=idx+1, col=col)
      fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=gt_y[i][:l+1].reshape(-1,), mode='markers+lines', marker=marker_dict_gt, name="{}-Y Ground Truth [{}]".format(set, i)), row=idx+1, col=col)
    if 'refinement' in args.pipeline:
      fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=y_refined[i][:l+1].reshape(-1,), mode='markers+lines', marker=marker_dict_refined, name="{}-Y Refined  [{}]".format(set, i)), row=idx+1, col=col)

    fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=y[i][:l+1].reshape(-1,), mode='markers+lines', marker=marker_dict_in_refnoisy, name="{}-Y Predicted [{}]".format(set, i)), row=idx+1, col=col)
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=flag[i][:l].reshape(-1,), mode='markers+lines', marker=marker_dict_pred, name="{}-EOT Predicted [{}]".format(set, i)), row=idx+1, col=col)

def loss_landscape_plot(loss_landscape, gt_dict, search_range, tid, all_opt=None):
  marker_dict_gt_height = dict(color='rgba(0, 255, 0, 1)', size=7)
  marker_dict_global_opt = dict(color='rgba(255, 0, 0, 1)', size=7)
  marker_dict_opt = dict(color='rgba(255, 127, 14, 1)', size=7)
  # Gravity marker
  marker_dict_pred_g = dict(color='rgba(0, 0, 255, 1)', size=4)
  marker_dict_mean_g = dict(color='rgba(255, 0, 0, 1)', size=4)
  marker_dict_std_g = dict(color='rgba(127, 0, 255, 1)', size=4)
  marker_dict_g = dict(color='rgba(0, 255, 0, 1.0)', size=4)
  fig_loss_landscape = make_subplots(rows=len(loss_landscape['loss'].keys()), cols=3, specs=[[{'type':'surface'}, {'type':'heatmap'}, {'type':'scatter'}]] * len(loss_landscape['loss'].keys()), horizontal_spacing=0.05, vertical_spacing=0.01)
  time_scale = (1/args.fps)**2
  for i, loss in enumerate(loss_landscape['loss'].keys()):
    if isinstance(loss_landscape['init_h']['first_h'], list):
      loss_landscape['init_h']['first_h'] = np.array(loss_landscape['init_h']['first_h']).reshape(-1)
    if isinstance(loss_landscape['init_h']['last_h'], list):
      loss_landscape['init_h']['last_h'] = np.array(loss_landscape['init_h']['last_h']).reshape(-1)

    x, y = search_range, search_range
    z = np.array(loss_landscape['loss'][loss]).reshape(search_range.shape[0], search_range.shape[0])
    global_min = np.where(np.isclose(z, np.min(z), rtol=1e-5))
    x_min, y_min = global_min[0][0], global_min[1][0]

    fig_loss_landscape.add_trace(go.Surface(x=x, y=y, z=z, name=loss, showscale=False), row=i+1, col=1)
    fig_loss_landscape.add_trace(go.Heatmap(x=x, y=y, z=z, name=loss, showscale=False), row=i+1, col=2)

    if gt_dict is not None:
      fh, lh = loss_landscape['init_h']['gt_h'][0].detach().cpu().numpy(), loss_landscape['init_h']['gt_h'][1].detach().cpu().numpy()
      fig_loss_landscape.add_trace(go.Scatter3d(x=lh, y=fh, z=loss_landscape['loss_gt'][loss], name="Ground truth height", mode='markers', marker=marker_dict_gt_height), row=i+1, col=1)
      #fig_loss_landscape.add_trace(go.Scatter3d(x=lh, y=fh, z=loss_landscape['loss_gt'][loss], name="Ground truth height", mode='markers', marker=marker_dict_gt_height), row=i+1, col=4)
      #fig_loss_landscape.add_trace(go.Scatter3d(x=loss_landscape['init_h']['last_h'], z=loss_landscape['loss'][loss], y=loss_landscape['init_h']['first_h'], mode='markers', marker=marker_dict_loss, name=loss), row=i+1, col=4)
      fig_loss_landscape.add_trace(go.Scatter3d(x=[search_range[y_min]], y=[search_range[x_min]], z=[z[x_min, y_min]], name="{} - Global Optimum".format(loss), mode='markers', marker=marker_dict_global_opt), row=i+1, col=1)
      #fig_loss_landscape.add_trace(go.Scatter3d(x=[search_range[y_min]], y=[search_range[x_min]], z=[z[x_min, y_min]], name="{} - Global Optimum".format(loss), mode='markers', marker=marker_dict_global_opt), row=i+1, col=4)

      fig_loss_landscape.add_trace(go.Scatter(x=lh, y=fh, name="Ground truth height", mode='markers', marker=marker_dict_gt_height), row=i+1, col=2)
      fig_loss_landscape.add_trace(go.Scatter(x=[search_range[y_min]], y=[search_range[x_min]], name="{} - Global Optimum".format(loss), mode='markers', marker=marker_dict_global_opt), row=i+1, col=2)

    # Visualize gravity along the diagonal
    c = len(search_range)
    for r in search_range:
      idx = int((c * r) + r)
      pred = loss_landscape['pred_dict'][idx]['xyz_refined'].detach().cpu().numpy()
      pred_1st_fd = pred[:, 1:, :] - pred[:, :-1, :]
      pred_accel = pred_1st_fd[:, 1:, :] - pred_1st_fd[:, :-1, :]
      pred_accel = pred_accel[..., [1]] / time_scale
      fig_loss_landscape.add_trace(go.Scatter(x=np.arange(pred_accel.shape[1]), y=pred_accel.reshape(-1), name="Predicted Gravity", mode='markers+lines', legendgroup=int(idx), marker=marker_dict_pred_g), row=i+1, col=3)
      fig_loss_landscape.add_trace(go.Scatter(x=np.arange(pred_accel.shape[1]), y=[np.mean(pred_accel)] * pred_accel.shape[1], name="Mean Predicted Gravity", mode='markers+lines', legendgroup=int(idx), marker=marker_dict_mean_g), row=i+1, col=3)
      fig_loss_landscape.add_trace(go.Scatter(x=np.arange(pred_accel.shape[1]), y=[np.mean(pred_accel) + np.std(pred_accel)] * pred_accel.shape[1], name="STD Predicted Gravity", mode='markers+lines', legendgroup=int(idx), marker=marker_dict_std_g), row=i+1, col=3)
      fig_loss_landscape.add_trace(go.Scatter(x=np.arange(pred_accel.shape[1]), y=[np.mean(pred_accel) - np.std(pred_accel)] * pred_accel.shape[1], name="STD Predicted Gravity", mode='markers+lines', legendgroup=int(idx), marker=marker_dict_std_g), row=i+1, col=3)
    fig_loss_landscape.add_trace(go.Scatter(x=np.arange(pred_accel.shape[1]), y=[-9.81] * pred_accel.shape[1], name="Gravity", mode='markers+lines', marker=marker_dict_g), row=i+1, col=3)
      
    if all_opt is not None:
      fh, lh = all_opt['init_h']['first_h'], all_opt['init_h']['last_h']
      z = all_opt['loss'][loss]
      fig_loss_landscape.add_trace(go.Scatter3d(x=lh, y=fh, z=z, name="Optimize - {}".format(i), mode='markers', marker=marker_dict_opt), row=i+1, col=1)
      fig_loss_landscape.add_trace(go.Scatter3d(x=lh, y=fh, z=z, name="Optimize - {}".format(i), mode='markers', marker=marker_dict_opt), row=i+1, col=3)


    fig_loss_landscape.update_scenes(xaxis_title_text="last_h (meter)", 
                                    yaxis_title_text="first_h (meter)", 
                                    zaxis_title_text="Loss")


  save_path = "{}_{}".format(args.config_yaml.split('/')[-2], args.save_suffix)
  utils_func.initialize_folder(path="{}/loss_landscape/{}/".format(args.vis_path, save_path))
  plotly.offline.plot(fig_loss_landscape, filename='{}/loss_landscape/{}/wandb_vis_traj_loss_landscape_{}.html'.format(args.vis_path, save_path, tid), auto_open=False)


def gravity_plot(pred_dict, gt_dict, tid):
  pred = pred_dict['xyz_refined']
  time_scale = (1/args.fps)**2

  # Finite diff 2 times : ds/dt -> dv/dt -> a
  pred_1st_fd = pred[:, 1:, :] - pred[:, :-1, :]
  #pred_1st_fd = gt[:, 1:, :] - gt[:, :-1, :]
  pred_accel = pred_1st_fd[:, 1:, :] - pred_1st_fd[:, :-1, :]

  gt = gt_dict['gt']
  # Finite diff 2 times : ds/dt -> dv/dt -> a
  gt_1st_fd = gt[:, 1:, :] - gt[:, :-1, :]
  #pred_1st_fd = gt[:, 1:, :] - gt[:, :-1, :]
  gt_accel = gt_1st_fd[:, 1:, :] - gt_1st_fd[:, :-1, :]

  fig_grav = go.Figure()
  fig_grav.add_trace(go.Scatter(x=np.arange(pred_accel.shape[1]), y=(pred_accel[..., [1]]/time_scale).detach().cpu().numpy().reshape(-1), name="Computed Gravity", mode='markers+lines'))
  fig_grav.add_trace(go.Scatter(x=np.arange(gt_accel.shape[1]), y=(gt_accel[..., [1]]/time_scale).detach().cpu().numpy().reshape(-1), name="Groundtruth Gravity", mode='markers+lines'))
  fig_grav.add_trace(go.Scatter(x=np.arange(pred_accel.shape[1]), y=[-9.81] * pred_accel.shape[1], name="Gravity", mode='markers+lines'))

  save_path = "{}_{}".format(args.config_yaml.split('/')[-2], args.save_suffix)
  utils_func.initialize_folder(path="{}/loss_landscape/{}/".format(args.vis_path, save_path))
  plotly.offline.plot(fig_grav, filename='{}/loss_landscape/{}/gravity_loss_landscape_{}.html'.format(args.vis_path, save_path, tid), auto_open=False)