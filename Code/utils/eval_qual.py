import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pred_path', type=str, default=None)
parser.add_argument('--pred_file', type=str, default=None)
parser.add_argument('--fps', type=float, required=True)
args = parser.parse_args()

def read_pred(f):
    if os.path.isfile(f):
        dat = np.load(f, allow_pickle=True)
        return dat.item()
    else: 
        return None

def distance(gt, pred, dist='l2'):
    if dist == 'l2':
        dist = np.sum((gt - pred)**2, axis=-1)
        return dist
    elif dist == 'l1':
        dist = np.sum(np.abs(gt - pred), axis=-1)
        return dist

def evaluate_g(dat):
    pred = []
    all_pts = []
    time_scale = (1/args.fps)**2
    g = 9.81
    g_vec = np.array([0, -g, 0])
    #g_vec = np.array([-g])

    threshold = [0.2, 0.4, 0.6, 0.8, 1, 5, 10, 20]
    error_count = {t:[] for t in threshold}
    for i in range(len(dat)):
        p = np.array(dat[i]['gt'])
        pred.append(p)
        
        v = p[1:] - p[:-1]
        a = v[1:] - v[:-1]
        a = a[..., [1]] / time_scale

        g_error = np.sqrt(distance(gt=g_vec, pred=a, dist='l2'))
        all_pts.append(a.shape[0])

        for t in threshold:
            below_g = g_error < + t*g
            #above_g = g_error > - t*g
            
            #count = np.sum(np.logical_and(below_g, above_g))
            count = np.sum(below_g)
            error_count[t].append(count)

    for t in threshold:
        print("Gravity error below threshold {} ({:.3f}, {:.3f}): {} from {}".format(t, g - t*g, g + t*g, np.sum(error_count[t]), np.sum(all_pts)))

def evaluate_belowground(dat):
    pred = []
    all_pts = []
    threshold = [0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1]
    error_count = {t:[] for t in threshold}
    for i in range(len(dat)):
        p = np.array(dat[i]['gt'])
        pred.append(p)
        y = p[..., [1]]
        all_pts.append(y.shape[0])
        for t in threshold:
            below_ground = y < -t
            
            count = np.sum(below_ground)
            error_count[t].append(count)

    for t in threshold:
        print("Ball drift below ground threshold {} m. : {} from {}".format(t, np.sum(error_count[t]), np.sum(all_pts)))

if __name__ == '__main__':
    if (args.pred_path is None and args.pred_file is None):
        raise FileNotFoundError
    
    file = args.pred_file
    dat = read_pred(file)
    evaluate_g(dat)
    evaluate_belowground(dat)


