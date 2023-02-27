import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import open3d as o3d
import argparse
import json
from models.vae import VAE
from caesarDataset import CaesarDataset
from tqdm import tqdm


folder = "caesar-fitted-meshes-pcd"
json = "data_subset.json"
genFolder = "test"

BATCH_SIZE = 20
EPOCHS = 10
LR = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_pcd(points, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)


def dataLoaders(folder, json, batch_size):
    trainDataset = CaesarDataset(folder, json, partition="train", seeds=1)
    testDataset = CaesarDataset(folder, json, partition="test", seeds=1)
    valDataset = CaesarDataset(folder, json, partition="val", seeds=1)
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=1, shuffle=True)
    valLoader = DataLoader(valDataset, batch_size=batch_size, shuffle=True)
    return trainLoader, testLoader, valLoader


def getModel():
    model = VAE()
    return model

def train(model, trainLoader, valLoader, epochs, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    minLoss = 1e10
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, (data, labels, names) in enumerate(tqdm(trainLoader)):
            gt = data[1]
            gt = torch.Tensor(gt).transpose(2, 1).float().to(device)
            optimizer.zero_grad()
            coarse, fine, mu, log_var = model(gt)
            loss = model.loss_function(gt, coarse, fine, mu, log_var)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(trainLoader)
        print("Epoch: {}, Train Loss: {}".format(epoch, train_loss))
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (data, labels, names) in enumerate(tqdm(valLoader)):
                gt = data[1]
                gt = torch.Tensor(gt).transpose(2, 1).float().to(device)
                optimizer.zero_grad()
                coarse, fine, mu, log_var = model(gt)
                loss = model.loss_function(gt, coarse, fine, mu, log_var)
                val_loss += loss.item()
        val_loss /= len(valLoader)
        print("Epoch: {}, Val Loss: {}".format(epoch, val_loss))

        if val_loss < minLoss:
            minLoss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved")

def test(model, testLoader, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, labels, names) in enumerate(tqdm(testLoader)):
            gt = data[1]
            gt = torch.Tensor(gt).transpose(2, 1).float().to(device)
            optimizer.zero_grad()
            coarse, fine, mu, log_var = model(gt)
            # print(fine.shape, gt.shape)
            loss = model.loss_function(gt, coarse, fine, mu, log_var)
            test_loss += loss.item()
            save_pcd(fine.squeeze().detach().cpu().numpy(), os.path.join(genFolder, "{}_fine.pcd".format(names[0])))
            save_pcd(data[1].squeeze(), os.path.join(genFolder, "{}_gt.pcd".format(names[0])))
    test_loss /= len(testLoader)
    print("Test Loss: {}".format(test_loss))

def generate(model, count, genFolder):
    model.to(device)
    if not os.path.exists(genFolder):
        os.makedirs(genFolder)

    for i in tqdm(range(count)):
        # Generate random latent vector with dimensions (6449, 3)
        z = torch.randn(1, 6449, 3)
        z = torch.Tensor(z).transpose(2, 1).float().to(device)
        out = model.generate(z)
        out = out.squeeze().detach().cpu().numpy()
        save_pcd(out, os.path.join(genFolder, "gen_{}.pcd".format(i)))


if __name__ == "__main__":
    trainLoader, testLoader, valLoader = dataLoaders(folder, json, batch_size=BATCH_SIZE)
    model = getModel()
    train(model, trainLoader, valLoader, epochs=EPOCHS, lr=LR)
    # Load Model
    model.load_state_dict(torch.load("best_model.pth"))
    # generate(model, 10, genFolder)
    test(model, testLoader, lr=LR)
    

