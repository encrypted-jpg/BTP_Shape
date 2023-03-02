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
from caesarDataset import CaesarDataset
from extensions.chamfer_dist import ChamferDistanceL1
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
import datetime
from utils.visual import plot_pcd_one_view
import random

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def log(fd,  message, time=True):
    if time:
        message = ' ==> '.join([datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), message])
    fd.write(message + '\n')
    fd.flush()
    print(message)

def prepare_logger(log_dir="log", exp_name="exp"):
    # prepare logger directory
    make_dir(log_dir)
    make_dir(os.path.join(log_dir, exp_name))

    logger_path = os.path.join(log_dir, exp_name)
    ckpt_dir = os.path.join(log_dir, exp_name, 'checkpoints')
    epochs_dir = os.path.join(log_dir, exp_name, 'epochs')

    make_dir(logger_path)
    make_dir(ckpt_dir)
    make_dir(epochs_dir)

    logger_file = os.path.join(log_dir, exp_name, 'logger.log')
    log_fd = open(logger_file, 'a')

    log(log_fd, "Experiment: {}".format(exp_name), False)
    log(log_fd, "Logger directory: {}".format(logger_path), False)

    train_writer = SummaryWriter(os.path.join(logger_path, 'train'))
    val_writer = SummaryWriter(os.path.join(logger_path, 'val'))

    return ckpt_dir, epochs_dir, log_fd, train_writer, val_writer

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

def getModel():
    return PCN(num_dense=6144, latent_dim=1024, grid_size=4)

def train(model, trainLoader, valLoader, epochs, lr, bestSavePath, lastSavePath):
    print("[+] Training the model...")
    # Get directory from the path
    dirPath = os.path.dirname(bestSavePath)
    ckpt_dir, epochs_dir, log_fd, train_writer, val_writer = prepare_logger(dirPath)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0)
    lr_schedual = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
    
    minLoss = 1e10
    minLossEpoch = 0
    train_step = 0
    val_step = 0
    chamfer = ChamferDistanceL1()
    for epoch in range(epochs):
        if train_step < 5000:
            alpha = 0.01
        elif train_step < 10000:
            alpha = 0.1
        elif train_step < 50000:
            alpha = 0.5
        else:
            alpha = 1.0
        
        model.train()
        train_loss = 0
        start = time.time()
        print("------------------Epoch: {}------------------".format(epoch))
        for i, (data, labels, names) in enumerate(tqdm(trainLoader)):
            gt = data[1]
            # gt = torch.Tensor(gt).transpose(2, 1).float().to(device)
            gt = gt.to(torch.float32).to(device)
            optimizer.zero_grad()
            coarse, fine = model(gt)
            loss1 = chamfer(coarse, gt)
            loss2 = chamfer(fine, gt)
            loss = loss1 + alpha * loss2
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            train_writer.add_scalar('loss', loss.item(), train_step)
            train_writer.add_scalar('coarse', loss1.item(), train_step)
            train_writer.add_scalar('dense', loss2.item(), train_step)
        train_loss /= len(trainLoader)
        end = time.time()
        # print("[+] Epoch: {}, Train Loss: {}, Time: {}".format(epoch, train_loss, end-start))
        lr_schedual.step()
        train_step += 1
        log(log_fd, "Epoch: {}, Train Loss: {}, Time: {}".format(epoch, train_loss, end-start))
        
        model.eval()
        val_loss = 0
        start = time.time()
        with torch.no_grad():
            rand_iter = random.randint(0, len(valLoader) - 1)

            for i, (data, labels, names) in enumerate(tqdm(valLoader)):
                gt = data[1]
                # gt = torch.Tensor(gt).transpose(2, 1).float().to(device)
                gt = gt.to(torch.float32).to(device)
                optimizer.zero_grad()
                coarse, fine = model(gt)
                loss1 = chamfer(coarse, gt)
                loss2 = chamfer(fine, gt)
                loss = loss1 + alpha * loss2
                val_loss += loss.item()

                if rand_iter == i:
                    index = random.randint(0, fine.shape[0] - 1)
                    plot_pcd_one_view(os.path.join(epochs_dir, 'epoch_{:03d}.png'.format(epoch)),
                                      [coarse[index].detach().cpu().numpy(), fine[index].detach().cpu().numpy(), gt[index].detach().cpu().numpy()],
                                      ['Coarse', 'Dense', 'Ground Truth'], xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))

        val_loss /= len(valLoader)
        end = time.time()
        # print("[+] Epoch: {}, Val Loss: {}, Time: {}".format(epoch, val_loss, end-start))
        val_step += 1
        val_writer.add_scalar('ValLoss', val_loss, val_step)
        log(log_fd, "Epoch: {}, Val Loss: {}, Time: {}".format(epoch, val_loss, end-start))

        if val_loss < minLoss:
            minLossEpoch = epoch
            minLoss = val_loss
            torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "loss": val_loss,
            }, bestSavePath)
            # print("[+] Best Model saved")
            log(log_fd, "Best Model saved")
        
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "loss": val_loss,
            }, lastSavePath)
        # print("[+] Last Model saved")
        log(log_fd, "Last Model saved")

    print("[+] Best Model saved at epoch: {} with loss {}".format(minLossEpoch, minLoss))
    log_fd.close()

def testModel(model, testLoader, testOut, lr, save):
    if not os.path.exists(testOut):
        os.makedirs(testOut)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.eval()
    test_loss = 0
    start = time.time()
    chamfer = ChamferDistanceL1()
    with torch.no_grad():
        for i, (data, labels, names) in enumerate(tqdm(testLoader)):
            gt = data[1]
            # gt = torch.Tensor(gt).transpose(2, 1).float().to(device)
            gt = gt.to(torch.float32).to(device)
            optimizer.zero_grad()
            coarse, fine = model(gt)
            loss1 = chamfer(coarse, gt)
            loss2 = chamfer(fine, gt)
            loss = loss1 + 0.01 * loss2
            test_loss += loss.item()
            if save:
                save_pcd(fine.squeeze().detach().cpu().numpy(), os.path.join(testOut, "{}_fine.pcd".format(names[0])))
                save_pcd(data[1].squeeze(), os.path.join(testOut, "{}_gt.pcd".format(names[0])))
    test_loss /= len(testLoader)
    end = time.time()
    print("[+] Test Loss: {}, Time: {}".format(test_loss, end-start))

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
        print("[+] Model Statistics - Epoch: {}, Loss: {}".format(checkpoint["epoch"], checkpoint["loss"]))
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
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
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
    
    model = getModel()

    if resume or test:
        model = load_model(model, modelPath)
    
    print(model)
    print("[+] Total Number of Parameters: {}".format(count_parameters(model)))
    
    if test:
        testModel(model, testLoader, testOut, lr=LR, save=testSave)
        exit()

    train(model, trainLoader, valLoader, epochs=EPOCHS, lr=LR, bestSavePath=bestSavePath, lastSavePath=lastSavePath)
    
    print("[+] Testing Model")
    testModel(model, testLoader, testOut, lr=LR, save=testSave)

    # Load Model
    model = load_model(model, bestSavePath)
    
    print("[+] Testing Model with best model")
    testModel(model, testLoader, testOut, lr=LR, save=testSave)
    
