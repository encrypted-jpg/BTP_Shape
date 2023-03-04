import os
import open3d as o3d
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json

class DFaustDataset(Dataset):
    def __init__(self, folder, jsonFile, partition="train", transform=None):
        self.folder = folder
        self.transform = transform
        self.partial = []
        self.gts = []
        self.labels = []
        count = 1
        with open(os.path.join(folder, jsonFile), 'r') as f:
            data = json.load(f)
        for name in tqdm(data[partition]):
            self.partial.append(name.replace("*", "partial"))
            self.gts.append(name.replace("*", "complete"))
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
        return (points, gt_points), np.array([self.labels[idx]]), os.path.basename(self.partial[idx]).replace(".pcd", "")