import torch as pt
import numpy as np
import argparse
from tqdm import tqdm
import glob, os
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):

  def __init__(self, dataset_path, trajectory_type):
    # Initialization
    '''
    self.dataset_path = {"Rolling" : glob.glob(dataset_path + "/Rolling*.npy"),
                         "MagnusProjectile" : glob.glob(dataset_path + "/MagnusProjectile*.npy"),
                         "Projectile" : glob.glob(dataset_path + "/Projectile*.npy"),
                         "Mixed" : glob.glob(dataset_path + "/Mixed*.npy")}
                         '''

    if dataset_path is None :
      print("[#] Cannot found dataset...Exit!!!")
      exit()
    elif len(os.listdir(dataset_path)) == 0:
      print("[#] Empty dataset folder...Exit!!!")
      exit()

    self.dataset_path = {trajectory_type : glob.glob(dataset_path + "/{}*.npy".format(trajectory_type))}
    self.trajectory_type = trajectory_type
    # Load data

    self.trajectory_dataset = {trajectory_type : [np.load(self.dataset_path[trajectory_type][i], allow_pickle=True) for i in tqdm(range(len(self.dataset_path[trajectory_type])), desc=trajectory_type, disable=True)]}
    '''
    self.trajectory_dataset = {"Rolling" : [np.load(self.dataset_path["Rolling"][i], allow_pickle=True) for i in tqdm(range(len(self.dataset_path["Rolling"])), desc="Rolling")],
                               "Projectile" : [np.load(self.dataset_path["Projectile"][i], allow_pickle=True) for i in tqdm(range(len(self.dataset_path["Projectile"])), desc="Projectile")],
                               "MagnusProjectile" : [np.load(self.dataset_path["MagnusProjectile"][i], allow_pickle=True) for i in tqdm(range(len(self.dataset_path["MagnusProjectile"])), desc="MagnusProjectile")],
                               "Mixed" : [np.load(self.dataset_path["Mixed"][i], allow_pickle=True) for i in tqdm(range(len(self.dataset_path["Mixed"])), desc="Mixed")]}
                               '''

    # Select trajectory type
    print("===============================Dataset shape===============================")
    for trajectory_type in self.trajectory_dataset.keys():
      self.trajectory_dataset[trajectory_type] = np.concatenate([self.trajectory_dataset[trajectory_type][i] for i in range(len(self.trajectory_dataset[trajectory_type]))])
      print("{} : {}".format(trajectory_type, self.trajectory_dataset[trajectory_type].shape))
    print("===========================================================================")

  def __len__(self):
    # Denotes the total number of samples
    return len(self.trajectory_dataset[self.trajectory_type])

  def __getitem__(self, idx):
    # Generates one batch of dataset by trajectory
    # print("At idx={} : {}".format(idx, self.trajectory_dataset[self.trajectory_type][idx].shape))
    # print(type(self.trajectory_dataset[self.trajectory_type]))
    return self.trajectory_dataset[self.trajectory_type][idx]

if __name__ == '__main__':
  print("************************************TESTING DATALOADER CLASS************************************")
  # For test the module
  parser = argparse.ArgumentParser(description='Trajectory dataloader')
  parser.add_argument('--dataset_path', dest='dataset_path', type=str, help='Specify path to dataset')
  parser.add_argument('--batch_size', type=int, help='Specify batch size', default=50)
  parser.add_argument('--trajectory_type', type=str, help="Specify trajectory type(Projectile, Rolling, MagnusProjectile)", default='Projectile')
  args = parser.parse_args()
  trajectory_dataset = TrajectoryDataset(args.dataset_path, trajectory_type=args.trajectory_type)
  trajectory_dataloader = DataLoader(trajectory_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False, collate_fn=collate_fn_padd, pin_memory=True)
  '''
  trajectory_val_dataset = TrajectoryDataset(args.dataset_path, trajectory_type=args.trajectory_type)
  trajectory_val_dataloader = DataLoader(trajectory_val_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False, collate_fn=collate_fn_padd, pin_memory=True)
  trajectory_val_iterloader = iter(trajectory_val_dataloader)
  for i in range(20):
    try:
      batch=next(trajectory_val_iterloader)
    except StopIteration:
      print("Reload a batch...")
      trajectory_val_iterloader = iter(trajectory_val_dataloader)
      batch=next(trajectory_val_iterloader)
    print(i, len(batch['input']))
  '''
  print("===============================Summary Batch (batch_size = {})===============================".format(args.batch_size))
  for key, batch in enumerate(trajectory_dataloader):
    print("Input batch [{}] : batch={}, lengths={}, mask={}".format(key, batch['input'][0].shape, batch['input'][1].shape, batch['input'][2].shape))
    print("Output batch [{}] : batch={}, lengths={}, mask={}".format(key, batch['output'][0].shape, batch['output'][1].shape, batch['output'][2].shape))

    # Test RNN/LSTM Step
    # 1.Pack the padded
    packed = pt.nn.utils.rnn.pack_padded_sequence(batch['input'][0], batch_first=True, lengths=batch['input'][1], enforce_sorted=False)
    # 2.RNN/LSTM model
    # 3.Unpack the packed
    unpacked = pt.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
    print("Unpacked equality : ", pt.eq(batch['input'][0], unpacked[0]).all())
    print("===========================================================================")
