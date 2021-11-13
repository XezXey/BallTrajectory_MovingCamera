import argparse
import os, glob, sys

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Input the model_ckpt path that contains config files', type=str, required=True)
parser.add_argument('--dataset_test_path', type=str, required=True)
parser.add_argument('--load_ckpt', type=str, required=True)
parser.add_argument('--save_cam_traj', type=str, default=None)
parser.add_argument('--recon', type=str, required=True)
parser.add_argument('--env', type=str, required=True)
parser.add_argument('--save_suffix', type=str, required=True)

# MISC
parser.add_argument('--batch_size', dest='batch_size', type=int, default=512)
parser.add_argument('--noise', dest='is_noise', action='store_true')
parser.add_argument('--no_noise', dest='is_noise', action='store_false')
parser.add_argument('--augment', dest='is_augment', action='store_true')
parser.add_argument('--no_augment', dest='is_augment', action='store_false')
parser.add_argument('--annealing', dest='is_annealing', action='store_true')
parser.add_argument('--no_annealing', dest='is_annealing', action='store_false')
parser.add_argument('--specific_dat', dest='specific_dat', type=str)
parser.add_argument('--fps', type=int, required=True)

args = parser.parse_args()

ckpt_path = glob.glob('{}/*'.format(args.config))

if args.is_noise:
    noise = '--noise'
else:
    noise = '--no_noise'

if args.is_annealing:
    annealing = '--annealing'
else:
    annealing = '--no_annealing'

if args.is_augment:
    augment = '--augment'
else:
    augment = '--no_augment'

for ckpt in ckpt_path:
    ckpt += '/config.yaml'
    if args.specific_dat not in ckpt:
        continue
    if args.save_cam_traj is not None:
        cmd = """
            python predict.py --dataset_test_path {} --load_ckpt {} --save_cam_traj {} --env {} {} {} {} --fps {} --recon {} --config {} --save_suffix {} --batch_size {} >> ../logs/log_{}_{}.txt
            """.format(args.dataset_test_path, args.load_ckpt, args.save_cam_traj, args.env,
            noise, annealing, augment, args.fps, args.recon, ckpt, args.save_suffix, args.batch_size, args.specific_dat, args.save_suffix)
    else:
        cmd = """
            python predict.py --dataset_test_path {} --load_ckpt {} --env {} {} {} {} --fps {} --recon {} --config {} --save_suffix {} --batch_size {} >> ../logs/log_{}_{}.txt
            """.format(args.dataset_test_path, args.load_ckpt, args.env,
            noise, annealing, augment, args.fps, args.recon, ckpt, args.save_suffix, args.batch_size, args.specific_dat, args.save_suffix)
    print("EXPERIMENT NAME : ", ckpt)
    try:
        os.system(cmd)
    except Exception:
        sys.exit()
