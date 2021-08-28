#%%
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import json
import plotly
import glob
import math

#%%
'''TRAINING DATA'''
# Features cols
features = ['x', 'y', 'z', 'u', 'v', 'd', 'intr_x', 'intr_y', 'intr_z', 'ray_x', 'ray_y', 'ray_z', 'cam_x', 'cam_y', 'cam_z', 
            'eot', 'og', 'rad', 'f_sin', 'f_cos', 'fx', 'fy', 'fz', 'fx_norm', 'fy_norm', 'fz_norm',
            'intrinsic', 'extrinsic', 'azimuth', 'elevation', 'extrinsic_inv', 'g']
x, y, z, u, v, d, intr_x, intr_y, intr_z, ray_x, ray_y, ray_z, cam_x, cam_y, cam_z, eot, og, rad, f_sin, f_cos, fx, fy, fz, fx_norm, fy_norm, fz_norm, intrinsic, extrinsic, azimuth, elevation, extrinsic_inv, g = range(len(features))
# Read training data
dataset = 'TracknetV1/bounce1_cont1_nowall/'
#dataset = 'Canonicalize'
train_path = '../../Dataset/{}/train_set/'.format(dataset)
train_dat = glob.glob(train_path + '*.npy')
tmp = []

for i in range(len(train_dat)):
    tmp.append(np.load(train_dat[i], allow_pickle=True))
train_dat = np.concatenate(tmp, axis=0)[..., ]

#%%
'''TESTING DATA'''
# Read testing data
tag = 'MovingCamera_TracknetV1_unbounce'
#tag = 'MovingCamera_CL'
name = 'dt_dt_cl_b2c2_nowall'
path = './tags_{}/{}'.format(tag, name)

print("DATA : ", path)
data = np.load(file='{}/reconstructed.npy'.format(path), allow_pickle=True)
pred = np.array(data[()]['pred'])
gt = np.array(data[()]['gt'])
cpos = np.array(data[()]['cpos'])

'''COLOR'''
def SetColor(x, r):
    if(x < r * 0.05):
        return "green"
    elif(x >= r*0.05 or x <= r*0.1):
        return "orange"
    elif(x > r*0.1):
        return "red"

'''PLOT'''
marker_dict_gt = dict(color='rgba(0, 0, 255, 0.4)', size=2)
marker_dict_noisy = dict(color='rgba(204, 102, 0, 0.7)', size=5)
marker_dict_cam = dict(color='rgba(180, 64, 16, 0.7)', size=5)
marker_dict_pred = dict(color='rgba(255, 0, 0, 0.4)', size=2)


#%%
# Plot all camera on training set
cx_train = []
cy_train = []
cz_train = []
for i in range(cpos.shape[0]):
    c_temp = np.unique(train_dat[i][..., [cam_x, cam_y, cam_z]].astype(np.float), axis=0)
    cx_train.append(c_temp[:, 0])
    cy_train.append(c_temp[:, 1])
    cz_train.append(c_temp[:, 2])

cx_train = np.array(cx_train).reshape(-1)
cy_train = np.array(cy_train).reshape(-1)
cz_train = np.array(cz_train).reshape(-1)


#%%
# Plot all camera on testing set
cx_test = []
cy_test = []
cz_test = []
rmse = []
error = []
for i in range(cpos.shape[0]):
    c_temp = np.unique(cpos[i][..., [0, 1, 2]].astype(np.float), axis=0)
    cx_test.append(c_temp[:, 0])
    cy_test.append(c_temp[:, 1])
    cz_test.append(c_temp[:, 2])
    #error_ = np.mean(np.sqrt(np.sum(((gt[i] - pred[i])**2), axis=-1)), axis=0)
    #print(np.sqrt(np.sum(((gt[i] - pred[i])**2), axis=-1)).shape)
    error_ = np.max(np.sqrt(np.sum(((gt[i] - pred[i])**2), axis=-1)), axis=0)
    error.append(error_)

cx_test = np.array(cx_test).reshape(gt.shape[0], -1)
cy_test = np.array(cy_test).reshape(gt.shape[0], -1)
cz_test = np.array(cz_test).reshape(gt.shape[0], -1)
print(cx_test.shape)

#%%
# All camera pose
#marker_dict_error = dict(color=error, colorscale='RdYlGn', size=2, reversescale=True)
#fig = go.Figure()
#fig.add_trace(go.Scatter3d(x=cx_train, y=cy_train, z=cz_train, mode='markers', marker=marker_dict_gt, name='Cam_train'))
#fig.add_trace(go.Scatter3d(x=cx_test, y=cy_test, z=cz_test, mode='markers', marker=marker_dict_error, name='Cam_test'))
#plotly.offline.plot(fig, filename='./{}/camera_variation.html'.format(path), auto_open=False)
#fig.show()
#
#
#plt.plot(error)
#plt.show()
#%%
def vis_traj_cam(idx, suffix):
    n = len(idx)
    fig = go.Figure()
    for i in idx:
        i = int(i)
        if ((cx_test[[i]].shape != cy_test[[i]].shape) 
            or (cx_test[[i]].shape != cz_test[[i]].shape) 
            or (cy_test[[i]].shape != cz_test[[i]].shape)):
            assert False
        else:
            for j in range(cx_test.shape[-1]):
                fig.add_trace(go.Scatter3d(x=cx_test[[i], j], y=cy_test[[i], j], z=cz_test[[i], j], mode='markers', marker=marker_dict_cam, name='Sample#{}'.format(i), legendgroup=i, legendgrouptitle_text=i))

        fig.add_trace(go.Scatter3d(x=pred[i][..., 0], y=pred[i][..., 1], z=pred[i][..., 2], mode='markers', marker=marker_dict_pred, name='Sample#{}-Pred'.format(i), legendgroup=i, legendgrouptitle_text=i, showlegend=False))
        fig.add_trace(go.Scatter3d(x=gt[i][..., 0], y=gt[i][..., 1], z=gt[i][..., 2], mode='markers', marker=marker_dict_gt, name='Sample#{}-Gt'.format(i), legendgroup=i, legendgrouptitle_text=i, showlegend=False))

    fig.add_trace(go.Scatter3d(x=cx_train, y=cy_train, z=cz_train, mode='markers', marker=marker_dict_gt, name='Cam_pos_train'))
    #plotly.offline.plot(fig, filename='./{}/prediction_{}-{}.html'.format(path, suffix, n), auto_open=False)
    fig.update_layout(title_text='Prediction_{}-{}.html'.format(suffix, n))
    fig.update_layout(scene=dict(aspectmode='cube'))
    fig.show()

# %%
# Trajectory with cam first-n
n = 10
idx = range(n)
vis_traj_cam(idx, suffix='first')

# %%
# Trajectory with cam random-n
fig = go.Figure()
n = 50
idx = np.random.randint(low=0, high=gt.shape[0], size=n)
vis_traj_cam(idx, suffix='rand')

# %%
# Trajectory with cam best/worst-n
n = 10
loss = []
for i in range(gt.shape[0]):
    x_l = np.sqrt(np.mean(gt[i][..., 0] - pred[i][..., 0])**2)
    y_l = np.sqrt(np.mean(gt[i][..., 1] - pred[i][..., 1])**2)
    z_l = np.sqrt(np.mean(gt[i][..., 2] - pred[i][..., 2])**2)
    l_ = np.sqrt(x_l**2 + y_l**2 + z_l**2)
    loss.append(l_)

loss = np.array(loss)
# Trajectory with cam worst-n
idx = np.argsort(loss)[::-1][:n]
vis_traj_cam(idx, suffix='worst')
# Trajectory with cam best-n
idx = np.argsort(loss)[:n]
vis_traj_cam(idx, suffix='best')
