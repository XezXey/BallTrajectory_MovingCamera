import os, sys, time
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
import utils.utils_func as utils_func
import utils.loss as utils_loss

# marker_dict for contain the marker properties
marker_dict_gt = dict(color='rgba(0, 0, 255, 0.4)', size=4)
marker_dict_noisy = dict(color='rgba(204, 102, 0, 0.4)', size=4)
marker_dict_pred = dict(color='rgba(255, 0, 0, 0.4)', size=4)
marker_dict_eot = dict(color='rgba(0, 255, 0, 0.4)', size=4)

def visualize_layout_update(fig=None, n_vis=3):
  # Save to html file and use wandb to log the html and display (Plotly3D is not working)
  for i in range(n_vis*2):
    #fig['layout']['scene{}'.format(i+1)].update(xaxis=dict(nticks=10, range=[-6, 6],), yaxis = dict(nticks=5, range=[-2, 2],), zaxis = dict(nticks=10, range=[-7, 7],), aspectmode='manual', aspectratio=dict(x=1, y=1, z=1))
    fig['layout']['scene{}'.format(i+1)].update(yaxis = dict(nticks=5, range=[-2, 6],),)
  return fig

def make_visualize(input_dict_train, gt_dict_train, input_dict_val, gt_dict_val, pred_dict_train, pred_dict_val, visualization_path, pred):
    # Visualize by make a subplots of trajectory
    n_vis = 5
    # Random the index the be visualize
    train_vis_idx = np.random.randint(low=0, high=input_dict_train['input'].shape[0], size=(n_vis))
    val_vis_idx = np.random.randint(low=0, high=input_dict_val['input'].shape[0], size=(n_vis))

    ####################################
    ############ Trajectory ############
    ####################################
    fig_traj = make_subplots(rows=n_vis, cols=2, specs=[[{'type':'scatter3d'}, {'type':'scatter3d'}]]*n_vis, horizontal_spacing=0.05, vertical_spacing=0.01)
    visualize_trajectory(pred=pred_dict_train['xyz'][..., [0, 1, 2]], gt=gt_dict_train['gt'][..., [0, 1, 2]], lengths=gt_dict_train['lengths'], mask=gt_dict_train['mask'][..., [0, 1, 2]], fig=fig_traj, flag='Train', vis_idx=train_vis_idx)
    visualize_trajectory(pred=pred_dict_val['xyz'][..., [0, 1, 2]], gt=gt_dict_val['gt'][..., [0, 1, 2]], lengths=gt_dict_val['lengths'], mask=gt_dict_val['mask'][..., [0, 1, 2]], fig=fig_traj, flag='Validation', vis_idx=val_vis_idx)

    fig_traj.update_layout(height=1920, width=1500, autosize=True) # Adjust the layout/axis for pitch scale
    fig_traj = visualize_layout_update(fig=fig_traj, n_vis=n_vis)

    #plotly.offline.plot(fig_traj, filename='./{}/trajectory_visualization_depth_pitch_scaled.html'.format(visualization_path), auto_open=False)
    plotly.offline.plot(fig_traj, filename='./{}/tracknetV1_g1c1.html'.format(visualization_path), auto_open=False)
    try:
      wandb.log({"PITCH SCALED : Trajectory Visualization(Col1=Train, Col2=Val)":wandb.Html(open('./{}/trajectory_visualization_depth_pitch_scaled.html'.format(visualization_path)))})
    except ValueError:
      print("[#] Wandb is not init")

    #plotly.offline.plot(fig_displacement, filename='./{}/trajectory_visualization_displacement.html'.format(visualization_path), auto_open=True)
    #wandb.log({"DISPLACEMENT VISUALIZATION":fig_displacement})

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

def visualize_trajectory(pred, gt, lengths, mask, vis_idx, fig=None, flag='Train'):
  # detach() for visualization
  pred = pred.cpu().detach().numpy()
  gt = gt.cpu().detach().numpy()
  # Training (Col1), Validation (Col2)
  col = 1 if flag == 'Train' else 2
  # Iterate to plot each trajectory
  for idx, i in enumerate(vis_idx):
    mse = utils_loss.TrajectoryLoss(pt.tensor(pred[i]).cuda(), pt.tensor(gt[i]).cuda(), mask=mask[i])
    fig.add_trace(go.Scatter3d(x=pred[i][:lengths[i], 0], y=pred[i][:lengths[i], 1], z=pred[i][:lengths[i], 2], mode='markers+lines', marker=marker_dict_pred, name="{}-Estimated Trajectory [{}], MSE = {:.3f}".format(flag, i, mse)), row=idx+1, col=col)
    fig.add_trace(go.Scatter3d(x=gt[i][:lengths[i], 0], y=gt[i][:lengths[i], 1], z=gt[i][:lengths[i], 2], mode='markers+lines', marker=marker_dict_gt, name="{}-Ground Truth Trajectory [{}]".format(flag, i)), row=idx+1, col=col)

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

def save_visualize(fig, postfix=None):
  if postfix is None:
    postfix = 0
  save_file_suffix = args.load_checkpoint.split('/')[-2]
  save_path = '{}/{}'.format(args.savetofile, save_file_suffix)
  utils_func.initialize_folder(save_path)
  plotly.offline.plot(fig, filename='./{}/interactive_optimize_{}.html'.format(save_path, postfix), auto_open=False)
