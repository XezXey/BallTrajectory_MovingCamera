import torch as pt
import numpy as np
import sys
import os
sys.path.append(os.path.realpath('../..'))
from sklearn.metrics import confusion_matrix
import plotly
import wandb
import utils.transformation as utils_transform
import matplotlib.pyplot as plt

# GPU initialization
if pt.cuda.is_available():
  device = pt.device('cuda')
else:
  device = pt.device('cpu')

args = None
def share_args(a):
  global args
  args = a

def ReprojectionLoss(pred, mask, lengths, cam_dict):
  # Reprojection loss
  u_pred, v_pred, _ = utils_transform.projection_2d(pts=pred, cam_dict=cam_dict)
  u_gt, v_gt = cam_dict['tracking'][..., [0]], cam_dict['tracking'][..., [1]]
  u_gt = u_gt * mask[..., [0]]
  v_gt = v_gt * mask[..., [0]]
  u_pred = u_pred * mask[..., [0]]
  v_pred = v_pred * mask[..., [0]]
  
  u_reprojection_loss = pt.sum((u_gt - u_pred)**2) / (pt.sum(mask[..., [0]]) + 1e-16)
  v_reprojection_loss = pt.sum((v_gt - v_pred)**2) / (pt.sum(mask[..., [0]]) + 1e-16)

  reprojection_loss = u_reprojection_loss + v_reprojection_loss
  return reprojection_loss

def GravityLoss(pred, gt, mask, lengths):
  # Compute the 2nd finite difference of the y-axis to get the gravity should be equal in every time step
  gravity_constraint_penalize = pt.tensor([0.])
  count = 0
  # Gaussian blur kernel for not get rid of the input information
  gaussian_blur = pt.tensor([0.25, 0.5, 0.25], dtype=pt.float32).view(1, 1, -1).to(device)
  # Kernel weight for performing a finite difference
  kernel_weight = pt.tensor([-1., 0., 1.], dtype=pt.float32).view(1, 1, -1).to(device)
  # Apply Gaussian blur and finite difference to gt
  for i in range(gt.shape[0]):
    # print(gt[i][:lengths[i]+1, 1])
    # print(gt[i][:lengths[i]+1, 1].shape)
    if gt[i][:lengths[i], 1].shape[0] < 6:
      print("The trajectory is too shorter to perform a convolution")
      continue
    gt_yaxis_1st_gaussian_blur = pt.nn.functional.conv1d(gt[i][:lengths[i], 1].view(1, 1, -1), gaussian_blur)
    gt_yaxis_1st_finite_difference = pt.nn.functional.conv1d(gt_yaxis_1st_gaussian_blur, kernel_weight)
    gt_yaxis_2nd_gaussian_blur = pt.nn.functional.conv1d(gt_yaxis_1st_finite_difference, gaussian_blur)
    gt_yaxis_2nd_finite_difference = pt.nn.functional.conv1d(gt_yaxis_2nd_gaussian_blur, kernel_weight)
    # Apply Gaussian blur and finite difference to gt
    pred_yaxis_1st_gaussian_blur = pt.nn.functional.conv1d(pred[i][:lengths[i], 1].view(1, 1, -1), gaussian_blur)
    pred_yaxis_1st_finite_difference = pt.nn.functional.conv1d(pred_yaxis_1st_gaussian_blur, kernel_weight)
    pred_yaxis_2nd_gaussian_blur = pt.nn.functional.conv1d(pred_yaxis_1st_finite_difference, gaussian_blur)
    pred_yaxis_2nd_finite_difference = pt.nn.functional.conv1d(pred_yaxis_2nd_gaussian_blur, kernel_weight)
    # Compute the penalize term
    # print(gt_yaxis_2nd_finite_difference, pred_yaxis_2nd_finite_difference)
    if gravity_constraint_penalize.shape[0] == 1:
      gravity_constraint_penalize = ((gt_yaxis_2nd_finite_difference - pred_yaxis_2nd_finite_difference)**2).reshape(-1, 1)
    else:
      gravity_constraint_penalize = pt.cat((gravity_constraint_penalize, ((gt_yaxis_2nd_finite_difference - pred_yaxis_2nd_finite_difference)**2).reshape(-1, 1)))

  return pt.mean(gravity_constraint_penalize)

def BelowGroundLoss(pred, mask, lengths):
  # Penalize when the y-axis is below on the ground
  pred = pred * mask
  below_ground_mask = pred[..., [1]] < 0
  below_ground_constraint_penalize = pt.mean((pred[..., [1]] * below_ground_mask)**2)
  return below_ground_constraint_penalize

def TrajectoryLoss(pred, gt, mask, lengths=None, delmask=True):
  # L2 loss of reconstructed trajectory
  x_trajectory_loss = pt.sum(((gt[..., 0] - pred[..., 0])**2) * mask[..., 0]) / pt.sum(mask[..., 0])
  y_trajectory_loss = pt.sum(((gt[..., 1] - pred[..., 1])**2) * mask[..., 1]) / pt.sum(mask[..., 1])
  z_trajectory_loss = pt.sum(((gt[..., 2] - pred[..., 2])**2) * mask[..., 2]) / pt.sum(mask[..., 2])

  return x_trajectory_loss + y_trajectory_loss + z_trajectory_loss

def EndOfTrajectoryLoss(pred, gt, mask, lengths):
  # Here we use output mask so we need to append the startpos to the pred before multiplied with mask(already included the startpos)
  pred = pred * mask
  gt = gt * mask
  # Implement from scratch
  # Flatten and concat all trajectory together
  gt = pt.cat(([gt[i][:lengths[i]] for i in range(lengths.shape[0])]))
  pred = pt.cat(([pred[i][:lengths[i]] for i in range(lengths.shape[0])]))
  # Class weight for imbalance class problem
  pos_weight = pt.sum(gt == 0)/(pt.sum(gt==1) + 1e-16)
  neg_weight = 1
  # Calculate the BCE loss
  eot_loss = pt.mean(-((pos_weight * gt * pt.log(pred + 1e-16)) + (neg_weight * (1-gt)*pt.log(1-pred + 1e-16))))

  return eot_loss

def eot_metrics_log(gt, pred, lengths, flag):
  pred = pred > 0.5
  # Output of confusion_matrix.ravel() = [TN, FP ,FN, TP]
  cm_each_trajectory = np.array([confusion_matrix(y_pred=pred[i][:lengths[i], :], y_true=gt[i][:lengths[i]]).ravel() for i in range(lengths.shape[0])])
  n_accepted_trajectory = np.sum(np.logical_and(cm_each_trajectory[:, 1]==0., cm_each_trajectory[:, 2] == 0.))
  cm_batch = np.sum(cm_each_trajectory, axis=0)
  tn, fp, fn, tp = cm_batch
  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  f1_score = 2 * (precision * recall) / (precision + recall)
  wandb.log({'{} Precision'.format(flag):precision, '{} Recall'.format(flag):recall, '{} F1-score'.format(flag):f1_score, '{}-#N accepted trajectory(Perfect EOT without FN, FP)'.format(flag):n_accepted_trajectory})

def LatentLoss(pred, gt, lengths, mask):
  '''
  Calculate the loss between reconstructed trajectory and the input latent
  input : 1) gt - col 3 contians eot flag, col 4: contains latent (depends on --features_cols)
          2) pred - trajectory xyz
  '''

  latent_loss = 0.0
  for i in range(pred.shape[0]):
    # for x in range(gt.shape[-1]):
      # print(x, gt[i][:, x])
    # exit()
    # eot = gt[i, :, 3]
    eot = gt[i, :, 9]
    # eot_location = pt.cat((pt.tensor([0]).to(device), pt.where(gt[i, :, 3] == 1)[0]))    # +1 for exclusive indexing
    eot_location = pt.cat((pt.tensor([0]).to(device), pt.where(gt[i, :, 9] == 1)[0]))    # +1 for exclusive indexing
    for j in range(eot_location.shape[0]-1):
      pred_j = pred[i, eot_location[j]:eot_location[j+1], [2, 0]] # sin-cos (x-z)
      latent_gt_j = gt[i, eot_location[j]:eot_location[j+1], [12, 13]]
      # plt.plot(pred_j[:, 0].detach().cpu().numpy())
      # plt.plot(pred_j[:, 1].detach().cpu().numpy())
      # plt.plot(gt[i, eot_location[j]:eot_location[j+1], 3].detach().cpu().numpy())
      # plt.show()
      pred_dt = pred_j[1:, :] - pred_j[:-1, :]
      pred_dt = pred_dt / (pt.sqrt(pt.sum(pred_dt**2, dim=-1, keepdims=True)) + 1e-16)
      if pt.unique(latent_gt_j).shape[0] > 2:
        latent_loss_j = 0.0
        print("[#] Shifted Latent")
        continue
      else:
        # print(pt.mean(latent_gt_j, dim=0, keepdims=True))
        latent_loss_j = pt.sum((pt.mean(latent_gt_j, dim=0, keepdims=True) - pred_dt) ** 2)
      latent_loss = latent_loss + latent_loss_j
    latent_loss = latent_loss / pred.shape[0]

  return latent_loss

def CosineSimLoss(pred, gt, input_dict, cam_dict, mask, lengths):
  '''
  Calculate the loss between reconstructed trajectory and the input latent
  input : 1) gt - col 3 contians eot flag, col 4: contains latent (depends on --features_cols)
          2) pred - trajectory xyz
  '''
  
  latent_loss = 0.0
  flag = input_dict['aux'][..., [0]]

  for i in range(pred.shape[0]):
    flag_ = flag[i][:, 0]
    flag_loc = pt.cat((pt.tensor([0]).to(device), pt.where(flag_ == 1)[0]))    # +1 for exclusive indexing
    for j in range(flag_loc.shape[0]-1):
      #dir_pred_j = pred[i][flag_loc[j]:flag_loc[j+2]+5, [1, 2]]
      #dir_gt_j = gt[i][flag_loc[j]:flag_loc[j+2]+5, [1, 2]]

      fig, axs = plt.subplots(3, sharex=True)
      dir_gt_j = gt[i][flag_loc[j]:flag_loc[j+1]+5, [1, 2]]
      axs[2].plot(dir_gt_j[:, 0].detach().cpu().numpy(), '-v', c='y')
      axs[2].plot(dir_gt_j[:, 1].detach().cpu().numpy(), '-v', c='y')

      dir_gt_j = gt[i][flag_loc[j]:flag_loc[j+1], [1, 2]]
      axs[2].plot(dir_gt_j[:, 0].detach().cpu().numpy(), '-v', c='g')
      axs[2].plot(dir_gt_j[:, 1].detach().cpu().numpy(), '-v', c='g')

      dir_pred_j = pred[i][flag_loc[j]:lengths[i], [1, 2]]
      dir_gt_j = gt[i][flag_loc[j]:lengths[i], [1, 2]]
      axs[0].plot(dir_pred_j[:, 0].detach().cpu().numpy(), '-x', c='r')
      axs[0].plot(dir_pred_j[:, 1].detach().cpu().numpy(), '-x', c='r')
      axs[1].plot(dir_gt_j[:, 0].detach().cpu().numpy(), '-o', c='b')
      axs[1].plot(dir_gt_j[:, 1].detach().cpu().numpy(), '-o', c='b')

      plt.savefig("{}/cosineSim.png".format(args.vis_path))
      plt.clf()
      input()
      # Normalized
      dir_pred = dir_pred_j[1:, :] - dir_pred_j[:-1, :]
      dir_pred = dir_pred / (pt.sqrt(pt.sum(dir_pred**2, dim=-1, keepdims=True)) + 1e-16)
      print(dir_pred.shape)
      print(dir_pred)
      latent_loss_j = pt.sum((pt.mean(dir_gt_j, dim=0, keepdims=True) - dir_pred) ** 2)

      latent_loss = latent_loss + latent_loss_j
    latent_loss = latent_loss / pred.shape[0]

  return latent_loss
