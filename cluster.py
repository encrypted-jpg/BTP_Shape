import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from models import PCN
from extensions.chamfer_dist import ChamferDistanceL1
from caesarDataset import CaesarDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
# Clustering Imports
from sklearn.cluster import KMeans



def getDataLoader(folder, json):
    trainDataset = CaesarDataset(folder, json, partition="train", seeds=1)
    trainLoader = DataLoader(trainDataset, batch_size=1, shuffle=True)
    return trainLoader

def getModel():
    return PCN(num_dense=6144, latent_dim=1024, grid_size=4)

def load_model(model, path):
    print("[+] Loading Model from: {}".format(path))
    checkpoint = torch.load(path)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("[+] Model Statistics - Epoch: {}, Loss: {}".format(checkpoint["epoch"], checkpoint["loss"]))
    except:
        model.load_state_dict(checkpoint)
    return model

def getReps(model, trainLoader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    reps = []
    for i, (data, labels, names) in enumerate(tqdm(trainLoader)):
        gt = data[1]
        # gt = torch.Tensor(gt).transpose(2, 1).float().to(device)
        gt = gt.to(torch.float32).to(device)
        rep = model.get_representation(gt).detach().cpu().numpy()
        reps.append(rep)
    reps = np.array(reps)
    return reps


def cluster(reps, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(reps)
    return kmeans.labels_


if __name__ == '__main__':
    folder = "caesar-fitted-meshes-pcd"
    json = "data.json"
    trainLoader = getDataLoader(folder, json)
    model = getModel()
    model = load_model("bestModel.pth")
    reps = getReps(model, trainLoader)
    labels = cluster(reps, 10)
    print(labels)
