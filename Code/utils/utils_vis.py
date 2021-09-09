import os, sys, time
sys.path.append(os.path.realpath('../'))
import numpy as np
import torch as pt
import json
import plotly
import math
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import wandb
import yaml
import utils.utils_func as utils_func
import utils.transformation as utils_transform
import utils.loss as utils_loss

# marker_dict for contain the marker properties
marker_dict_gt = dict(color='rgba(0, 0, 255, 0.4)', size=4)
marker_dict_noisy = dict(color='rgba(204, 102, 0, 0.4)', size=4)
marker_dict_pred = dict(color='rgba(255, 0, 0, 0.4)', size=4)
marker_dict_eot = dict(color='rgba(0, 255, 0, 0.4)', size=4)
marker_dict_cam = dict(color='rgba(255, 0, 0, 0.4)', size=10)

args = None
def share_args(a):
  global args
  args = a

def visualize_layout_update(fig=None, n_vis=5):
  # Save to html file and use wandb to log the html and display (Plotly3D is not working)
  for i in range(n_vis*2):
    #fig['layout']['scene{}'.format(i+1)].update(xaxis=dict(nticks=10, range=[-6, 6],), yaxis = dict(nticks=5, range=[-2, 2],), zaxis = dict(nticks=10, range=[-7, 7],), aspectmode='manual', aspectratio=dict(x=1, y=1, z=1))
    fig['layout']['scene{}'.format(i+1)].update(yaxis = dict(nticks=5, range=[-2, 6],),)
  return fig

def wandb_vis(input_dict_train, gt_dict_train, pred_dict_train, cam_dict_train,
                  input_dict_val, gt_dict_val, pred_dict_val, cam_dict_val):
    '''
    Make a visualization for logging a prediction to wandb
    Input : 
      1. {input, gt, pred, cam}_dict_train
      2. {input, gt, pred, cam}_dict_val
    '''

    # Visualize by make a subplots of trajectory
    n_vis = 5
    # Random the index the be visualize
    train_vis_idx = np.random.choice(np.arange(input_dict_train['input'].shape[0]), size=(n_vis), replace=False)
    val_vis_idx = np.random.choice(np.arange(input_dict_val['input'].shape[0]), size=(n_vis), replace=False)

    ####################################
    ############ Trajectory ############
    ####################################
    fig_traj = make_subplots(rows=n_vis, cols=2, specs=[[{'type':'scatter3d'}, {'type':'scatter3d'}]]*n_vis, horizontal_spacing=0.05, vertical_spacing=0.01)
    visualize_trajectory(pred=pred_dict_train['xyz'][..., [0, 1, 2]], gt=gt_dict_train['gt'][..., [0, 1, 2]], lengths=gt_dict_train['lengths'], mask=gt_dict_train['mask'][..., [0, 1, 2]], fig=fig_traj, set='Train', vis_idx=train_vis_idx, col=1)
    visualize_trajectory(pred=pred_dict_val['xyz'][..., [0, 1, 2]], gt=gt_dict_val['gt'][..., [0, 1, 2]], lengths=gt_dict_val['lengths'], mask=gt_dict_val['mask'][..., [0, 1, 2]], fig=fig_traj, set='Validation', vis_idx=val_vis_idx, col=2)
    ####################################
    ############### Ray ################
    ####################################
    visualize_ray(cam_dict=cam_dict_train, input_dict=input_dict_train, fig=fig_traj, set='Train', vis_idx=train_vis_idx, col=1)
    visualize_ray(cam_dict=cam_dict_val, input_dict=input_dict_val, fig=fig_traj, set='Val', vis_idx=val_vis_idx, col=2)

    fig_traj.update_layout(height=1920, width=1500, autosize=True) # Adjust the layout/axis for pitch scale

    plotly.offline.plot(fig_traj, filename='{}/wandb_vis.html'.format(args.vis_path), auto_open=False)
    try:
      wandb.log({"Trajectory(Col1=Train, Col2=Val)":wandb.Html(open('{}/wandb_vis.html'.format(args.vis_path)))})
    except ValueError:
      print("[#] Wandb is not init")

def inference_vis(input_dict, gt_dict, pred_dict, cam_dict):
    n_vis = 10 if input_dict['input'].shape[0] > 10 else input_dict['input'].shape[0]
    vis_idx = np.random.choice(np.arange(input_dict['input'].shape[0]), size=(n_vis), replace=False)
    # Variables
    pred = pred_dict['xyz'][..., [0, 1, 2]]
    len_ = input_dict['lengths']
    mask = input_dict['mask'][..., [0, 1, 2]]
    if not args.no_gt:
      gt = gt_dict['gt'][..., [0, 1, 2]]
    else:
      gt = None
    ####################################
    ############ Trajectory ############
    ####################################
    fig_traj = make_subplots(rows=math.ceil(n_vis/2), cols=2, specs=[[{'type':'scatter3d'}, {'type':'scatter3d'}]]*math.ceil(n_vis/2), horizontal_spacing=0.05, vertical_spacing=0.01)
    visualize_trajectory(pred=pred, gt=gt, lengths=len_, mask=mask, fig=fig_traj, set='Test', vis_idx=vis_idx[:n_vis//2], col=1)
    visualize_trajectory(pred=pred, gt=gt, lengths=len_, mask=mask, fig=fig_traj, set='Test', vis_idx=vis_idx[n_vis//2:], col=2)
    ####################################
    ############### Ray ################
    ####################################
    visualize_ray(cam_dict=cam_dict, input_dict=input_dict, fig=fig_traj, set='Test', vis_idx=vis_idx[:n_vis//2], col=1)
    visualize_ray(cam_dict=cam_dict, input_dict=input_dict, fig=fig_traj, set='Test', vis_idx=vis_idx[n_vis//2:], col=2)

    fig_traj.update_layout(height=1920, width=1500, autosize=True) # Adjust the layout/axis for pitch scale

    plotly.offline.plot(fig_traj, filename='{}/pred_vis.html'.format(args.vis_path), auto_open=False)


def visualize_ray(cam_dict, input_dict, fig, set, vis_idx, col):
  # Iterate to plot each trajectory
  #ray = cam_dict['ray'].cpu().numpy()
  ray = utils_transform.cast_ray(uv=cam_dict['tracking'], I=cam_dict['I'], E=cam_dict['E'])
  intr = utils_transform.ray_to_plane(E=cam_dict['E'], ray=ray)
  cam_pos = np.linalg.inv(cam_dict['E'].cpu().numpy())[..., 0:3, -1]
  len_ = input_dict['lengths']

  for idx, i in enumerate(vis_idx):
    draw_ray_x = list()
    draw_ray_y = list()
    draw_ray_z = list()
    len_temp = len_[i]
    c_temp = cam_pos[i][:len_temp]
    r_temp = ray[i][:len_temp]
    intr_temp = intr[i][:len_temp]

    for j in range(c_temp.shape[0]):
        # Cam
        draw_ray_x.append(c_temp[j, 0])
        draw_ray_y.append(c_temp[j, 1])
        draw_ray_z.append(c_temp[j, 2])
        # Ray
        draw_ray_x.append(intr_temp[j, 0])
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
        mode='lines',
        line = dict(width = 0.2, color = 'rgba(0, 255, 0, 0.7)'),
        name='{}-Ray-Trajectory [{}]'.format(set, i).format(i), 
        legendgroup=int(i)
    ), row=idx+1, col=col)
  
    c_temp = np.unique(c_temp, axis=0)
    fig.add_trace(go.Scatter3d(
        x=c_temp[..., 0],
        y=c_temp[..., 1],
        z=c_temp[..., 2],
        mode='markers',
        marker=marker_dict_cam,
        name='{}-Cam-Trajectory [{}]'.format(set, i).format(i), 
        legendgroup=int(i)
    ), row=idx+1, col=col)

  return fig

'''
def visualize_ray(cam_dict, input_dict, fig, set, vis_idx, col):
  # Iterate to plot each trajectory
  #ray = cam_dict['ray'].cpu().numpy()
  ray = utils_transform.cast_ray(uv=cam_dict['tracking'], I=cam_dict['I'], E=cam_dict['E'])
  cam_pos = np.linalg.inv(cam_dict['E'].cpu().numpy())[..., 0:3, -1]
  len_ = input_dict['lengths']

  for idx, i in enumerate(vis_idx):
    draw_ray_x = list()
    draw_ray_y = list()
    draw_ray_z = list()
    len_temp = len_[i]
    c_temp = cam_pos[i][:len_temp]
    r_temp = ray[i][:len_temp]

    for j in range(c_temp.shape[0]):
        # Cam
        draw_ray_x.append(c_temp[j, 0])
        draw_ray_y.append(c_temp[j, 1])
        draw_ray_z.append(c_temp[j, 2])
        # Ray
        draw_ray_x.append(c_temp[j, 0] + r_temp[j, 0]*70)
        draw_ray_y.append(c_temp[j, 1] + r_temp[j, 1]*70)
        draw_ray_z.append(c_temp[j, 2] + r_temp[j, 2]*70)
        # No-interconnect
        draw_ray_x.append(None)
        draw_ray_y.append(None)
        draw_ray_z.append(None)

    ## set the mode to lines to plot only the lines and not the balls/markers
    fig.add_trace(go.Scatter3d(
        x=draw_ray_x,
        y=draw_ray_y,
        z=draw_ray_z,
        mode='lines',
        line = dict(width = 0.7, color = 'rgba(0, 255, 0, 0.7)'),
        name='{}-Ray-Trajectory [{}]'.format(set, i).format(i), 
        legendgroup=int(i)
    ), row=idx+1, col=col)
  
    c_temp = np.unique(c_temp, axis=0)
    fig.add_trace(go.Scatter3d(
        x=c_temp[..., 0],
        y=c_temp[..., 1],
        z=c_temp[..., 2],
        mode='markers',
        marker=marker_dict_cam,
        name='{}-Cam-Trajectory [{}]'.format(set, i).format(i), 
        legendgroup=int(i)
    ), row=idx+1, col=col)

  return fig
'''

def visualize_displacement(input_dict, pred_dict, gt_dict, pred_eot, gt_eot, vis_idx, pred, cam_params_dict, fig=None, flag='train', n_vis=5):
  duv = pred_dict['input'][..., [0, 1]].cpu().detach().numpy()
  if 'depth' in args.pipeline:
    depth = pred_dict[pred].cpu().detach().numpy()
  else: depth = None
  lengths = input_dict['lengths'].cpu().detach().numpy()
  if pred_eot is not None:
    pred_eot = pred_eot.cpu().detach().numpy()
  if gt_eot is not None:
    gt_eot = gt_eot.cpu().detach().numpy()
  # Change the columns for each set
  if flag == 'Train': col = 1
  elif flag == 'Validation': col=2
  # Iterate to plot each trajectory
  for idx, i in enumerate(vis_idx):
    ####################################
    ############## DEPTH ###############
    ####################################
    if pred=='depth' and 'depth' in args.pipeline:
      if args.bi_pred_avg or args.bi_pred_ramp or args.bi_pred_weight:
        fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=depth[i][:lengths[i], 0], mode='markers+lines', marker=marker_dict_pred, name='{}-traj#{}-Displacement of Forward DEPTH'.format(flag, i)), row=idx+1, col=col)
        fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=depth[i][:lengths[i], 1], mode='markers+lines', marker=marker_dict_pred, name='{}-traj#{}-Displacement of Backward DEPTH'.format(flag, i)), row=idx+1, col=col)
      else:
        fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=depth[i][:lengths[i], 0], mode='markers+lines', marker=marker_dict_pred, name='{}-traj#{}-Displacement of DEPTH'.format(flag, i)), row=idx+1, col=col)

    ####################################
    ############## dU, dV ##############
    ####################################
    if 'uv' in args.pipeline:

      uv_gt = np.cumsum(pt.cat((input_dict['startpos'][..., [0, 1]], input_dict['input'][..., [0, 1]]), dim=1).cpu().detach().numpy(), axis=1)
      uv_pred = np.cumsum(pt.cat((input_dict['startpos'][..., [0, 1]], pred_dict['pred_uv'][..., [0, 1]]), dim=1).cpu().detach().numpy(), axis=1)

      duv_gt = input_dict['input'][..., [0, 1]].cpu().detach().numpy()
      duv_pred = pred_dict['pred_uv'][..., [0, 1]].cpu().detach().numpy()

      if i in pred_dict['missing_idx']:
        nan_idx = np.where(pred_dict['missing_mask'][i].cpu().numpy()==True)[0]
        duv[i][nan_idx, :] = np.nan
      fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=duv[i][:lengths[i], 0], mode='markers+lines', marker=marker_dict_noisy, name='{}-traj#{}-Input dU'.format(flag, i)), row=idx+1, col=col)
      fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=duv[i][:lengths[i], 1], mode='markers+lines', marker=marker_dict_noisy, name='{}-traj#{}-Input dV'.format(flag, i)), row=idx+1, col=col)
      fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=duv_pred[i][:lengths[i], 0], mode='markers+lines', marker=marker_dict_pred, name='{}-traj#{}-Interpolated dU'.format(flag, i)), row=idx+1, col=col)
      fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=duv_pred[i][:lengths[i], 1], mode='markers+lines', marker=marker_dict_pred, name='{}-traj#{}-Interpolated dV'.format(flag, i)), row=idx+1, col=col)
      fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=duv_gt[i][:lengths[i], 0], mode='markers+lines', marker=marker_dict_gt, name='{}-traj#{}-Ground Truth dU'.format(flag, i)), row=idx+1, col=col)
      fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=duv_gt[i][:lengths[i], 1], mode='markers+lines', marker=marker_dict_gt, name='{}-traj#{}-Ground Truth dV'.format(flag, i)), row=idx+1, col=col)


      fig.add_trace(go.Scatter(x=uv_pred[i][:lengths[i]+1, 0], y=uv_pred[i][:lengths[i]+1, 1], mode='markers+lines', marker=marker_dict_pred, name='{}-traj#{}-Interpolated UV'.format(flag, i)), row=idx+1, col=col)

      fig.add_trace(go.Scatter(x=uv_gt[i][:lengths[i]+1, 0], y=uv_gt[i][:lengths[i]+1, 1], mode='markers+lines', marker=marker_dict_gt, name='{}-traj#{}-Ground Truth UV'.format(flag, i)), row=idx+1, col=col)

    else:
      world_pred = pt.unsqueeze(pred_dict['finale_xyz'][i, :, [0, 1, 2]], dim=0)
      world_pred = utils_transform.centering(xyz=world_pred, cam_params_dict=cam_params_dict, device=device, rev=True)
      world_gt = pt.unsqueeze(gt_dict['xyz'][i, :, [0, 1, 2]], dim=0)
      world_gt = utils_transform.centering(xyz=world_gt, cam_params_dict=cam_params_dict, device=device, rev=True)
      u_pred_proj, v_pred_proj, d_pred_proj = utils_transform.projectToScreenSpace(world=world_pred, cam_params_dict=cam_params_dict['main'])
      uv_pred_proj = pt.cat((u_pred_proj, v_pred_proj), dim=2).cpu().detach().numpy()
      u_gt_proj, v_gt_proj, d_gt_proj = utils_transform.projectToScreenSpace(world=world_gt, cam_params_dict=cam_params_dict['main'])
      uv_gt_proj = pt.cat((u_gt_proj, v_gt_proj), dim=2).cpu().detach().numpy()

      fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=duv[i][:lengths[i], 0], mode='markers+lines', marker=marker_dict_gt, name='{}-traj#{}-Displacement of U'.format(flag, i)), row=idx+1, col=col)
      fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=duv[i][:lengths[i], 1], mode='markers+lines', marker=marker_dict_gt, name='{}-traj#{}-Displacement of V'.format(flag, i)), row=idx+1, col=col)

      fig.add_trace(go.Scatter(x=uv_pred_proj[0][:lengths[i], 0], y=uv_pred_proj[0][:lengths[i], 1], mode='markers+lines', marker=marker_dict_pred, name='{}-traj#{}-UV_Pred_Projection'.format(flag, i)), row=idx+1, col=col)
      fig.add_trace(go.Scatter(x=uv_gt_proj[0][:lengths[i], 0], y=uv_gt_proj[0][:lengths[i], 1], mode='markers+lines', marker=marker_dict_gt, name='{}-traj#{}-UV_Gt'.format(flag, i)), row=idx+1, col=col)

    ####################################
    ############### EOT ################
    ####################################
    if pred_eot is not None:
      fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=pred_eot[i][:lengths[i], 0], mode='markers+lines', marker=marker_dict_pred, name='{}-traj#{}-EOT(Pred)'.format(flag, i)), row=idx+1, col=col)
    if gt_eot is not None:
      fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=gt_eot[i][:lengths[i], 0], mode='markers+lines', marker=marker_dict_eot, name='{}-traj#{}-EOT(GT)'.format(flag, i)), row=idx+1, col=col)

def visualize_trajectory(pred, gt, lengths, mask, vis_idx, set, col, fig=None):
  pred = pred.cpu().detach().numpy()
  if gt is not None:
    gt = gt.cpu().detach().numpy()
  # Iterate to plot each trajectory
  for idx, i in enumerate(vis_idx):
    if gt is not None:
      mse = utils_loss.TrajectoryLoss(pt.tensor(pred[i]).cuda(), pt.tensor(gt[i]).cuda(), mask=mask[i])
      fig.add_trace(go.Scatter3d(x=gt[i][:lengths[i], 0], y=gt[i][:lengths[i], 1], z=gt[i][:lengths[i], 2], mode='markers+lines', 
                                marker=marker_dict_gt, name="{}-Ground Truth Trajectory [{}], MSE={:3f}".format(set, i, mse)), row=idx+1, col=col)

    fig.add_trace(go.Scatter3d(x=pred[i][:lengths[i], 0], y=pred[i][:lengths[i], 1], z=pred[i][:lengths[i], 2], mode='markers+lines', 
                              marker=marker_dict_pred, name="{}-Estimated Trajectory [{}]".format(set, i)), row=idx+1, col=col)

def visualize_eot(pred, gt, startpos, lengths, mask, vis_idx, fig=None, flag='Train', n_vis=5):
  # pred : concat with startpos and stack back to (batch_size, sequence_length+1, 1)
  pred = pt.stack([pt.cat([startpos[i], pred[i]]) for i in range(startpos.shape[0])])
  gt = pt.stack([pt.cat([startpos[i], gt[i]]) for i in range(startpos.shape[0])])
  # Here we use output mask so we need to append the startpos to the pred before multiplied with mask(already included the startpos)
  pred *= mask
  gt *= mask
  # Weight of positive/negative classes for imbalanced class
  pos_weight = pt.sum(gt == 0)/pt.sum(gt==1)
  neg_weight = 1
  eps = 1e-10
  # Calculate the EOT loss for each trajectory
  eot_loss = pt.mean(-((pos_weight * gt * pt.log(pred+eps)) + (neg_weight * (1-gt)*pt.log(1-pred+eps))), dim=1).cpu().detach().numpy()

  # detach() for visualization
  pred = pred.cpu().detach().numpy()
  gt = gt.cpu().detach().numpy()
  lengths = lengths.cpu().detach().numpy()
  # Change the columns for each set
  if flag == 'Train': col = 1
  elif flag == 'Validation': col=2
  # Iterate to plot each trajectory
  for idx, i in enumerate(vis_idx):
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]).reshape(-1,), y=pred[i][:lengths[i], :].reshape(-1,), mode='markers+lines', marker=marker_dict_pred, name="{}-EOT Predicted [{}], EOTLoss = {:.3f}".format(flag, i, eot_loss[i][0])), row=idx+1, col=col)
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]).reshape(-1,), y=gt[i][:lengths[i], :].reshape(-1,), mode='markers+lines', marker=marker_dict_gt, name="{}-Ground Truth EOT [{}]".format(flag, i)), row=idx+1, col=col)