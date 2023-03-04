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
import pickle


def getDataLoader(folder, json):
    trainDataset = CaesarDataset(folder, json, partition="train", seeds=1)
    trainLoader = DataLoader(trainDataset, batch_size=1, shuffle=True)
    return trainLoader

def getTestDataLoader(folder, json):
    testDataset = CaesarDataset(folder, json, partition="test", seeds=1)
    testLoader = DataLoader(testDataset, batch_size=1)
    return testLoader

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
    gts = []
    for i, (data, labels, names) in enumerate(tqdm(trainLoader)):
        gt = data[1]
        # gt = torch.Tensor(gt).transpose(2, 1).float().to(device)
        gt = gt.to(torch.float32).to(device)
        rep = model.get_representation(gt).detach().cpu().numpy()
        reps.append(rep)
        gts.append(data[1][0].numpy().reshape(-1, 3))
    reps = np.array(reps)
    gts = np.array(gts)
    return reps, gts

def cluster(reps, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(reps)
    return kmeans

def load_clusters(path):
    modelDict = pickle.load(open(path, "rb"))
    kmeans = modelDict["kmeans"]
    top5pcs = modelDict["top5pcs"]
    return kmeans, top5pcs

def getNearest(pc, top5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    chamfer = ChamferDistanceL1()
    dists = []
    for i in range(len(top5)):
        cmp = torch.Tensor(np.array([top5[i]])).to(torch.float32).to(device)
        dist = chamfer(pc, cmp)
        dists.append(dist.item())
    return top5[np.argmin(dists)]

def getNPC(pc, model, kmeans, top5pcs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    gt = torch.Tensor(np.array([pc])).to(torch.float32).to(device)
    rep = model.get_representation(gt).detach().cpu().numpy()
    label = kmeans.predict(rep)[0]
    top5 = top5pcs[label]
    nearestPC = getNearest(gt, top5)
    return nearestPC

def getD(pc1, pc2):
    chamfer = ChamferDistanceL1()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pc1 = torch.Tensor(np.array([pc1])).to(torch.float32).to(device)
    pc2 = torch.Tensor(np.array([pc2])).to(torch.float32).to(device)
    return chamfer(pc1, pc2).item()


if __name__ == '__main__':
    folder = "caesar-fitted-meshes-pcd"
    json = "data.json"
    trainLoader = getDataLoader(folder, json)
    testLoader = getTestDataLoader(folder, json)
    model = getModel()
    model = load_model(model, "bestModel.pth")
    reps, gts = getReps(model, testLoader)
    # kmeans = cluster(reps, 10)
    # print(kmeans.labels_)

    kmeans, top5pcs = load_clusters("modelDict.pkl")
    chamfer = ChamferDistanceL1()
    pc = gts[0]
    sim = getNPC(pc, model, kmeans, top5pcs)
    print(pc.shape, sim.shape)
    print(getD(pc, sim))
    reps1, gts1 = getReps(model, trainLoader)
    dists = []
    for i in range(len(gts1)):
        dists.append(getD(gts1[i], pc))
    print(min(dists))
