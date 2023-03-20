import os
import open3d as o3d
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json

class CaesarDataset(Dataset):
    """
    Folder has the following structure:
    caesar-fitted-meshes-pcd
    |-- train
    |   |-- complete
    |   |   |-- *.pcd
    |   |-- partial
    |       |-- *
    |           |-- *-[1-8].pcd
    |-- test
    |   |-- complete
    |   |   |-- *.pcd
    |   |-- partial
    |       |-- *
    |           |-- *-[1-8].pcd
    |-- val
        |-- complete
        |   |-- *.pcd
        |-- partial
            |-- *
                |-- *-[1-8].pcd
    Use the json file to get the list of files to be used for training, testing and validation.
    """
    def __init__(self, folder, jsonFile, partition="train", seeds=8, gt_num_points=6144, p_num_points=384, transform=None):
        self.folder = folder
        self.transform = transform
        self.partial = []
        self.gts = []
        self.labels = []
        self.gt_num_points = gt_num_points
        self.seeds = list(range(1, seeds + 1))
        count = 1
        with open(os.path.join(folder, jsonFile), 'r') as f:
            data = json.load(f)
        for name in tqdm(data[partition]):
            for seed in self.seeds:
                self.partial.append(os.path.join(folder, partition, "partial", name, name + "-" + str(seed) + ".pcd"))
                self.gts.append(os.path.join(folder, partition, "complete", name + ".pcd"))
                self.labels.append(count)
            count += 1

    def __len__(self):
        return len(self.partial)
    def __getitem__(self, idx):
        pcd = o3d.io.read_point_cloud(self.partial[idx])
        points = np.asarray(pcd.points)
        if self.transform:
            points = self.transform(points)
        gt = o3d.io.read_point_cloud(self.gts[idx])
        gt_points = np.asarray(gt.points)
        gt_points = gt_points[np.random.choice(gt_points.shape[0], self.gt_num_points, replace=False), :]
        return (points, gt_points), np.array([self.labels[idx]]), os.path.basename(self.partial[idx]).replace(".pcd", "")