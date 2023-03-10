import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import open3d as o3d
import argparse
import json
from models import *
from datasets.caesarDataset import CaesarDataset
from tqdm import tqdm
import time


def save_pcd(points, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def dataLoaders(folder, json, batch_size):
    print("[+] Loading the data...")
    trainDataset = CaesarDataset(folder, json, partition="train", seeds=1)
    testDataset = CaesarDataset(folder, json, partition="test", seeds=1)
    valDataset = CaesarDataset(folder, json, partition="val", seeds=1)
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=1, shuffle=True)
    valLoader = DataLoader(valDataset, batch_size=batch_size, shuffle=True)
    return trainLoader, testLoader, valLoader


def getModel(modelName):
    modelName = modelName.lower()
    lossWeights = [1, 1, 1]
    if modelName == "vae":
        model = VAE()
        lossWeights = [2, 5, 2]
    elif modelName == "sphericalvae":
        model = SphericalVAE()
    elif modelName == "foldingvae":
        model = FoldingVAE(6449)
        lossWeights = [1, 1, 1]
    else:
        raise Exception("[-] Model not found!")
    return model, lossWeights

def train(model, trainLoader, valLoader, epochs, lr, bestSavePath, lastSavePath, lossWeights=[1, 1, 1]):
    print("[+] Training the model...")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    minLoss = 1e10
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_chamfer_loss = 0
        train_kld_loss = 0
        start = time.time()
        print("------------------Epoch: {}------------------".format(epoch))
        for i, (data, labels, names) in enumerate(tqdm(trainLoader)):
            gt = data[1]
            gt = torch.Tensor(gt).transpose(2, 1).float().to(device)
            optimizer.zero_grad()
            coarse, fine, mu, log_var = model(gt)
            loss, floss, kld = model.loss_function(gt, coarse, fine, mu, log_var, weight=lossWeights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_chamfer_loss += floss.item()
            train_kld_loss += kld.item()
        train_loss /= len(trainLoader)
        train_chamfer_loss /= len(trainLoader)
        train_kld_loss /= len(trainLoader)
        end = time.time()
        print("[+] Epoch: {}, Train Loss: {}, Chamfer Loss: {}, KLD Loss: {}, Time: {}".format(epoch, train_loss, train_chamfer_loss, train_kld_loss, end-start))

        
        model.eval()
        val_loss = 0
        val_chamfer_loss = 0
        val_kld_loss = 0
        start = time.time()
        with torch.no_grad():
            for i, (data, labels, names) in enumerate(tqdm(valLoader)):
                gt = data[1]
                gt = torch.Tensor(gt).transpose(2, 1).float().to(device)
                optimizer.zero_grad()
                coarse, fine, mu, log_var = model(gt)
                loss, floss, kld = model.loss_function(gt, coarse, fine, mu, log_var, weight=lossWeights)
                val_loss += loss.item()
                val_chamfer_loss += floss.item()
                val_kld_loss += kld.item()
        val_loss /= len(valLoader)
        val_chamfer_loss /= len(valLoader)
        val_kld_loss /= len(valLoader)
        end = time.time()
        print("[+] Epoch: {}, Val Loss: {}, Chamfer Loss: {}, KLD Loss: {}, Time: {}".format(epoch, val_loss, val_chamfer_loss, val_kld_loss, end-start))

        if val_loss < minLoss:
            minLoss = val_loss
            torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "loss": val_loss,
            "chamfer_loss": val_chamfer_loss,
            "kld_loss": val_kld_loss,
            }, bestSavePath)
            print("[+] Best Model saved")
        
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "loss": val_loss,
            "chamfer_loss": val_chamfer_loss,
            "kld_loss": val_kld_loss,
            }, lastSavePath)
        print("[+] Last Model saved")

def testModel(model, testLoader, testOut, lr, save, lossWeights=[1, 1, 1]):
    if not os.path.exists(testOut):
        os.makedirs(testOut)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.eval()
    test_loss = 0
    test_chamfer_loss = 0
    test_kld_loss = 0
    start = time.time()
    with torch.no_grad():
        for i, (data, labels, names) in enumerate(tqdm(testLoader)):
            gt = data[1]
            gt = torch.Tensor(gt).transpose(2, 1).float().to(device)
            optimizer.zero_grad()
            coarse, fine, mu, log_var = model(gt)
            loss, floss, kld = model.loss_function(gt, coarse, fine, mu, log_var)
            test_loss += loss.item()
            test_chamfer_loss += floss.item()
            test_kld_loss += kld.item()
            if save:
                save_pcd(fine.squeeze().detach().cpu().numpy(), os.path.join(testOut, "{}_fine.pcd".format(names[0])))
                save_pcd(data[1].squeeze(), os.path.join(testOut, "{}_gt.pcd".format(names[0])))
    test_loss /= len(testLoader)
    test_chamfer_loss /= len(testLoader)
    test_kld_loss /= len(testLoader)
    end = time.time()
    print("[+] Test Loss: {}, Chamfer Loss: {}, KLD Loss: {}, Time: {}".format(test_loss, test_chamfer_loss, test_kld_loss, end-start))

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

def load_model(model, path):
    print("[+] Loading Model from: {}".format(path))
    checkpoint = torch.load(path)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("[+] Model Statistics - Epoch: {}, Loss: {}, Chamfer Loss: {}, KLD Loss: {}".format(checkpoint["epoch"], checkpoint["loss"], checkpoint["chamfer_loss"], checkpoint["kld_loss"]))
    except:
        model.load_state_dict(checkpoint)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="caesar-fitted-meshes-pcd", help="Path to dataset")
    parser.add_argument("--json", type=str, default="data_subset.json", help="Path to json file")
    parser.add_argument("--genFolder", type=str, default="gen", help="Path to generated files")
    parser.add_argument("--testOut", type=str, default="testOut", help="Path to test output")
    parser.add_argument("--savePath", type=str, default=".", help="Path to save model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--model", type=str, default="bestModel.pth", help="Path to model")
    parser.add_argument("--test", action="store_true", help="Test model")
    parser.add_argument("--testSave", action="store_true", help="Save test output")
    args = parser.parse_args()
    
    folder = args.folder
    json = args.json
    genFolder = args.genFolder
    testOut = args.testOut
    bestSavePath = os.path.join(args.savePath, "bestModel.pth")
    lastSavePath = os.path.join(args.savePath, "lastModel.pth")
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    resume = args.resume
    modelPath = args.model
    test = args.test
    testSave = args.testSave

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trainLoader, testLoader, valLoader = dataLoaders(folder, json, batch_size=BATCH_SIZE)
    
    model, lossWeights = getModel("foldingvae")

    if resume or test:
        model = load_model(model, modelPath)
    
    print(model)
    print("[+] Total Number of Parameters: {}".format(count_parameters(model)))
    
    if test:
        testModel(model, testLoader, testOut, lr=LR, save=testSave, lossWeights=lossWeights)
        exit()

    train(model, trainLoader, valLoader, epochs=EPOCHS, lr=LR, bestSavePath=bestSavePath, lastSavePath=lastSavePath, lossWeights=lossWeights)
    
    print("[+] Testing Model")
    testModel(model, testLoader, testOut, lr=LR, save=testSave, lossWeights=lossWeights)

    # Load Model
    model = load_model(model, bestSavePath)
    
    print("[+] Testing Model with best model")
    testModel(model, testLoader, testOut, lr=LR, save=testSave, lossWeights=lossWeights)
    
