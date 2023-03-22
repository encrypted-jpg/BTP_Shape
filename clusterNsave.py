import torch
import open3d as o3d
import numpy as np
import os
import json
import argparse
from models import *
from datasets.dfaustDataset import DFaustDataset
from datasets.caesarDataset import CaesarDataset
from datasets.scapeDataset import ScapeDataset
from torch.utils.data import DataLoader
from extensions.chamfer_dist import ChamferDistanceL1
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Point Cloud Completion Clustering')
    parser.add_argument('--folder', type=str, default='caesar-fitted-meshes-pcd', help='dfaust, caesar, scape')
    parser.add_argument('--json' , type=str, default='data.json')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--pcnModelPath', type=str, default='checkpoints/caesarBestModel.pth', help='path to pcn model')
    parser.add_argument('--clusterPath', type=str, default='checkpoints/caesarModelDict384.pkl', help='path to cluster model')
    return parser.parse_args()

def dataLoaders(args):
    print("[+] Loading the data...")
    folder = args.folder
    json = args.json
    if "caesar" in folder.lower():
        seeds = 8
        trainDataset = CaesarDataset(folder, json, partition="train", seeds=seeds)
        testDataset = CaesarDataset(folder, json, partition="test", seeds=seeds)
        valDataset = CaesarDataset(folder, json, partition="val", seeds=seeds)
    elif "d-faust" in folder.lower():
        trainDataset = DFaustDataset(folder, json, partition="train")
        testDataset = DFaustDataset(folder, json, partition="test")
        valDataset = DFaustDataset(folder, json, partition="val")
    elif "scape" in folder.lower():
        seeds = 16
        trainDataset = ScapeDataset(folder, json, partition="train", seeds=seeds)
        testDataset = ScapeDataset(folder, json, partition="test", seeds=seeds)
        valDataset = ScapeDataset(folder, json, partition="val", seeds=seeds)
    trainLoader = DataLoader(trainDataset, batch_size=1, shuffle=False)
    testLoader = DataLoader(testDataset, batch_size=1, shuffle=False)
    valLoader = DataLoader(valDataset, batch_size=1, shuffle=False)
    return trainLoader, testLoader, valLoader

def main():
    args = parse_args()
    trainLoader, testLoader, valLoader = dataLoaders(args)
    chamfer = ChamferDistanceL1()
    cluster = Cluster(args.pcnModelPath, args.clusterPath, chamfer)
    pdict = {}
    for i, (data, labels, names) in enumerate(tqdm(trainLoader)):
        par, gt = data
        par = par.cuda()
        pred = cluster.getIdx(par)
        for j in range(len(pred)):
            pdict[str(names[j]).replace("\\", "/")] = pred[j]
            # print(names[j], pdict[names[j]])
    
    for i, (data, labels, names) in enumerate(tqdm(testLoader)):
        par, gt = data
        par = par.cuda()
        pred = cluster.getIdx(par)
        for j in range(len(pred)):
            pdict[str(names[j]).replace("\\", "/")] = pred[j]
            # print(names[j], pdict[names[j]])
    
    for i, (data, labels, names) in enumerate(tqdm(valLoader)):
        par, gt = data
        par = par.cuda()
        pred = cluster.getIdx(par)
        for j in range(len(pred)):
            pdict[str(names[j]).replace("\\", "/")] = pred[j]
            # print(names[j], pdict[names[j]])
    
    with open("caesar_cluster.json", "w") as f:
        json.dump(pdict, f, indent=4)

if __name__ == "__main__":
    main()
