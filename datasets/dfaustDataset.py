import os
import open3d as o3d
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json

class DFaustDataset(Dataset):
    def __init__(self, folder, jsonFile, partition="train", gt_num_points=6144, transform=None):
        self.folder = folder
        self.transform = transform
        self.gt_num_points = gt_num_points
        self.partial = []
        self.gts = []
        self.labels = []
        count = 1
        jfolder = folder
        if "D-Faust" not in folder:
            jfolder = os.path.join(folder, "D-Faust")
        elif "D-Faust-Reg" in folder:
            folder = folder.replace("D-Faust-Reg", "")
        elif "D-Faust" in folder:
            folder = folder.replace("D-Faust", "")
        with open(os.path.join(jfolder, jsonFile), 'r') as f:
            data = json.load(f)
        for name in tqdm(data[partition]):
            self.partial.append(os.path.join(folder, name.replace("*", "partial")))
            self.gts.append(os.path.join(folder, name.replace("*", "complete")))
            self.labels.append(count)
            count += 1
        self.cache = {}
        self.cacheLen = 10000

    def __len__(self):
        return len(self.partial)
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        pcd = o3d.io.read_point_cloud(self.partial[idx])
        points = np.asarray(pcd.points)
        if self.transform:
            points = self.transform(points)
        gt = o3d.io.read_point_cloud(self.gts[idx])
        gt_points = np.asarray(gt.points)
        gt_points = gt_points[np.random.choice(gt_points.shape[0], self.gt_num_points, replace=False), :]
        name = self.partial[idx][self.partial[idx].find("D-Faust"):]
        if len(self.cache) < self.cacheLen:
            self.cache[idx] = (points, gt_points), np.array([self.labels[idx]]), name.replace(".pcd", "")
        return (points, gt_points), np.array([self.labels[idx]]), name.replace(".pcd", "")