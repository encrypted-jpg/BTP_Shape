import os
from datasets.caesarDataset import CaesarDataset
from datasets.dfaustDataset import DFaustDataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

folder = "D-Faust"

BATCH_SIZE = 8

trainDataset = DFaustDataset(folder, "data_subset.json", partition="train")
testDataset = DFaustDataset(folder, "data_subset.json", partition="test")
valDataset = DFaustDataset(folder, "data_subset.json", partition="val")
trainLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
testLoader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=True)
valLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=True)

for i, (data, labels, names) in enumerate(tqdm(trainLoader)):
    print(data[0].shape, data[1].shape, labels.shape)
    print(labels)
    print(names)
    break





