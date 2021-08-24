import torch as pt
import numpy as np
import sys
import os
sys.path.append(os.path.realpath('../..'))
from sklearn.metrics import confusion_matrix
import plotly
import wandb
import utils.transformation as transformation
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

def ReprojectionLoss(pred, gt, mask, lengths, cam_params_dict, normalize):
  u_pred, v_pred, _ = transformation.projectToScreenSpace(pred, cam_params_dict, normalize)
  u_gt, v_gt, _ = transformation.projectToScreenSpace(gt, cam_params_dict, normalize)
  u_reprojection_loss = (pt.sum((((u_gt[..., 0] - u_pred[..., 0]))**2) * mask[..., 0]) / pt.sum(mask[..., 0]))
  v_reprojection_loss = (pt.sum((((v_gt[..., 0] - v_pred[..., 0]))**2) * mask[..., 0]) / pt.sum(mask[..., 0]))
  return (u_reprojection_loss + v_reprojection_loss)/2


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

def BelowGroundPenalize(pred, gt, mask, lengths):
  # Penalize when the y-axis is below on the ground
  pred = pred * mask
  below_ground_mask = pred[..., 1] < 0
  below_ground_constraint_penalize = pt.mean((pred[..., 1] * below_ground_mask)**2)
  return below_ground_constraint_penalize

def TrajectoryLoss(pred, gt, mask, lengths=None, delmask=True):
  # L2 loss of reconstructed trajectory
  x_trajectory_loss = pt.sum(((gt[..., 0] - pred[..., 0])**2) * mask[..., 0]) / pt.sum(mask[..., 0])
  y_trajectory_loss = pt.sum(((gt[..., 1] - pred[..., 1])**2) * mask[..., 1]) / pt.sum(mask[..., 1])
  z_trajectory_loss = pt.sum(((gt[..., 2] - pred[..., 2])**2) * mask[..., 2]) / pt.sum(mask[..., 2])
  print(x_trajectory_loss)
  print(y_trajectory_loss)
  print(z_trajectory_loss)

  #x_trajectory_loss = (pt.sum((((gt[:, 3:, 0] - pred[:, 3:, 0]))**2) * mask[:, 3:, 0]) / pt.sum(mask[:, 3:, 0]))
  #y_trajectory_loss = (pt.sum((((gt[:, 3:, 1] - pred[:, 3:, 1]))**2) * mask[:, 3:, 1]) / pt.sum(mask[:, 3:, 1]))
  #z_trajectory_loss = (pt.sum((((gt[:, 3:, 2] - pred[:, 3:, 2]))**2) * mask[:, 3:, 2]) / pt.sum(mask[:, 3:, 2]))
  # weight_loss = create_weight_loss(mask)
  # x_trajectory_loss = (pt.sum((((gt[..., 0] - pred[..., 0]))**2) * weight_loss[..., 0] * mask[..., 0]) / pt.sum(mask[..., 0]))
  # y_trajectory_loss = (pt.sum((((gt[..., 1] - pred[..., 1]))**2) * weight_loss[..., 1] * mask[..., 1]) / pt.sum(mask[..., 1]))
  # z_trajectory_loss = (pt.sum((((gt[..., 2] - pred[..., 2]))**2) * weight_loss[..., 0] * mask[..., 2]) / pt.sum(mask[..., 2]))
  return x_trajectory_loss + y_trajectory_loss + z_trajectory_loss

def create_weight_loss(mask):
  if len(mask.shape)==2:
    # Handle the visualization that have shape (seq_len, 3) but this function accept (batch_size, seq_len, 3)
    mask = pt.unsqueeze(mask, dim=0)

  weight_loss = pt.zeros(mask.shape).to(device)
  for idx, length in enumerate(pt.sum(mask, dim=1)[..., [0]]):
    weight_loss[idx][:length[0], [0, 1, 2]] = pt.linspace(10, 1, length[0]).view(-1, 1).to(device)
  # print(weight_loss[0][:pt.sum(mask, dim=1)[..., [0]][0]+2, [0]])
  # print(mask[0][:pt.sum(mask, dim=1)[..., [0]][0]+2, [0]])
  return weight_loss

def DepthLoss(pred, gt, mask, lengths):
  depth_loss = (pt.sum((((gt - pred))**2) * mask) / pt.sum(mask))
  return depth_loss

def EndOfTrajectoryLoss(pred, gt, startpos, mask, lengths, flag='Train'):
  # Here we use output mask so we need to append the startpos to the pred before multiplied with mask(already included the startpos)
  pred = pred * mask
  gt = gt * mask
  # Prevent of pt.log(-value)
  eps = 1e-10
  # Implement from scratch
  # Flatten and concat all trajectory together
  gt = pt.cat(([gt[i][:lengths[i]+1] for i in range(startpos.shape[0])]))
  pred = pt.cat(([pred[i][:lengths[i]+1] for i in range(startpos.shape[0])]))
  # Class weight for imbalance class problem
  pos_weight = pt.sum(gt == 0)/(pt.sum(gt==1) + eps)
  neg_weight = 1
  # Calculate the BCE loss
  eot_loss = pt.mean(-((pos_weight * gt * pt.log(pred + eps)) + (neg_weight * (1-gt)*pt.log(1-pred + eps))))

  # Log the precision, recall, confusion_matrix and using wandb
  # gt_log = gt.clone().cpu().detach().numpy()
  # pred_log = pred.clone().cpu().detach().numpy()
  # eot_metrics_log(gt=gt_log, pred=pred_log, lengths=lengths.cpu().detach().numpy(), flag=flag)

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

def MultiviewReprojectionLoss(pred, gt, mask, lengths, cam_params_dict):
  all_multiview_loss = 0

  # De-centering
  pred = transformation.centering(xyz=pred, cam_params_dict=cam_params_dict, device=device, rev=True)
  gt = transformation.centering(xyz=gt, cam_params_dict=cam_params_dict, device=device, rev=True)

  pred = pred * mask
  gt = gt * mask
  # print(pred[0][:lengths[0]+2])
  # print(gt[0][:lengths[0]+2])
  # print(mask[0][:lengths[0]+2])
  # exit()

  for cam_use in args.multiview_loss:
    u_gt, v_gt, d_gt = transformation.projectToScreenSpace(world=gt, cam_params_dict=cam_params_dict[cam_use], normalize=False)
    u_pred, v_pred, d_pred = transformation.projectToScreenSpace(world=pred, cam_params_dict=cam_params_dict[cam_use], normalize=False)
    # print(u_pred[..., 0])
    # print(u_pred[..., 0] * mask[..., 1])
    # print(u_pred[..., 0] * mask[..., 1] * mask[..., 1])
    # exit()
    u_reprojection_loss = (pt.sum((((u_gt[..., 0] - u_pred[..., 0]) * d_gt[..., 0])**2) * mask[..., 0]) / pt.sum(mask[..., 0]))
    v_reprojection_loss = (pt.sum((((v_gt[..., 0] - v_pred[..., 0]) * d_gt[..., 0])**2) * mask[..., 1]) / pt.sum(mask[..., 1]))
    multiview_loss = (u_reprojection_loss + v_reprojection_loss)/2
    all_multiview_loss += multiview_loss

  return all_multiview_loss

def InterpolationLoss(uv_gt, uv_pred, mask, lengths):
  u_interpolation_loss = (pt.sum((((uv_gt[..., 0] - uv_pred[..., 0]))**2) * mask[..., 0]) / pt.sum(mask[..., 0]))
  v_interpolation_loss = (pt.sum((((uv_gt[..., 1] - uv_pred[..., 1]))**2) * mask[..., 0]) / pt.sum(mask[..., 0]))
  interpolation_loss = u_interpolation_loss + v_interpolation_loss
  return interpolation_loss

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
