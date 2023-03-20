import os
from datasets.caesarDataset import CaesarDataset
from datasets.dfaustDataset import DFaustDataset
from datasets.scapeDataset import ScapeDataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

folder = "scape-pcd"

BATCH_SIZE = 8

trainDataset = ScapeDataset(folder, "data.json", partition="train")
testDataset = ScapeDataset(folder, "data.json", partition="test")
valDataset = ScapeDataset(folder, "data.json", partition="val")
trainLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
testLoader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=True)
valLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=True)

for i, (data, labels, names) in enumerate(tqdm(trainLoader)):
    print(data[0].shape, data[1].shape, labels.shape)
    print(labels)
    print(names)
    break





