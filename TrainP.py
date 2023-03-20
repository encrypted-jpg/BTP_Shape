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
from datasets.dfaustDataset import DFaustDataset
from datasets.scapeDataset import ScapeDataset
from extensions.chamfer_dist import ChamferDistanceL1
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
import datetime
from utils.visual import plot_pcd_one_view
import random
import visdom
from knn import kNNLoss

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def print_log(fd,  message, time=True):
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

    print_log(log_fd, "Experiment: {}".format(exp_name), False)
    print_log(log_fd, "Logger directory: {}".format(logger_path), False)

    train_writer = SummaryWriter(os.path.join(logger_path, 'train'))
    val_writer = SummaryWriter(os.path.join(logger_path, 'val'))

    return ckpt_dir, epochs_dir, log_fd, train_writer, val_writer

def save_pcd(points, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def dataLoaders(args):
    print("[+] Loading the data...")
    folder = args.folder
    json = args.json
    batch_size = args.batch_size
    if "caesar" in folder.lower():
        seeds = 8 if args.partial else 1
        trainDataset = CaesarDataset(folder, json, partition="train", seeds=seeds)
        testDataset = CaesarDataset(folder, json, partition="test", seeds=seeds)
        valDataset = CaesarDataset(folder, json, partition="val", seeds=seeds)
    elif "d-faust" in folder.lower():
        trainDataset = DFaustDataset(folder, json, partition="train")
        testDataset = DFaustDataset(folder, json, partition="test")
        valDataset = DFaustDataset(folder, json, partition="val")
    elif "scape" in folder.lower():
        seeds = 16 if args.partial else 1
        trainDataset = ScapeDataset(folder, json, partition="train", seeds=seeds)
        testDataset = ScapeDataset(folder, json, partition="test", seeds=seeds)
        valDataset = ScapeDataset(folder, json, partition="val", seeds=seeds)
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=1, shuffle=True)
    valLoader = DataLoader(valDataset, batch_size=batch_size, shuffle=True)
    return trainLoader, testLoader, valLoader

def getModel():
    return PCN(num_dense=6144, latent_dim=1024, grid_size=4)

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def scheduler_to(sched, device):
    for param in sched.__dict__.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)

def load_model_train(model, modelPath, optimizer, lr_scheduler):
    print("[+] Loading the model...")
    checkpoint = torch.load(modelPath, map_location="cuda")
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        # if "optimizer_state_dict" in checkpoint.keys():
        #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #     optimizer_to(optimizer, "cuda")
        #     print("[+] Optimizer loaded")
        # if "lr_scheduler_state_dict" in checkpoint.keys():
        #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        #     scheduler_to(lr_scheduler, "cuda")
        #     print("[+] LR Scheduler loaded")
        if "epoch" in checkpoint.keys():
            epoch = checkpoint['epoch']
        else:
            epoch = 0
        if "loss" in checkpoint.keys():
            loss = checkpoint['loss']
        else:
            loss = 1e10
        if "loss_log" in checkpoint.keys():
            loss_log = checkpoint["loss_log"]
        else:
            loss_log = {"train": []}
    except:
        model.load_state_dict(checkpoint)
        epoch = 0
        loss = 1e10
        loss_log = {"train": []}
    return model, optimizer, lr_scheduler, epoch, loss, loss_log

def get_visdom(port):
    vis = visdom.Visdom(port=args.visdom)
    color_num = 4
    chunk_size = int(6144 / color_num)
    colors = np.array([(227,0,27),(231,64,28),(237,120,15),(246,176,44),
                        (252,234,0),(224,221,128),(142,188,40),(18,126,68),
                        (63,174,0),(113,169,156),(164,194,184),(51,186,216),
                        (0,152,206),(16,68,151),(57,64,139),(96,72,132),
                        (172,113,161),(202,174,199),(145,35,132),(201,47,133),
                        (229,0,123),(225,106,112),(163,38,42),(128,128,128)])
    colors = colors[np.random.choice(len(colors), color_num, replace=False)]
    label = torch.stack([torch.ones(chunk_size).type(torch.LongTensor) * inx for inx in range(1,int(color_num)+1)], dim=0).view(-1)
    return vis, colors, label

def train(model, trainLoader, valLoader, args):
    print("[+] Training the model...")
    if args.partial:
        print("[+] Using PARTIAL POINT CLOUDS AS INPUT")
    # Get directory from the path
    bestSavePath = os.path.join(args.savePath, "bestModel.pth")
    lastSavePath = os.path.join(args.savePath, "lastModel.pth")
    dirPath = os.path.dirname(bestSavePath)
    ckpt_dir, epochs_dir, log_fd, train_writer, val_writer = prepare_logger(dirPath)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vis, colors, label = get_visdom(args.visdom)
    print_log(log_fd, "Visdom Connection: {}".format(vis.check_connection()))

    minLossEpoch = 0
    train_step = 0
    val_step = 0
    loss_log = {"train": []}
    epoch = 0
    if args.resume:
        model, optimizer, lr_scheduler, epoch, minLoss, loss_log = load_model_train(model, args.modelPath, optimizer, lr_scheduler)
        train_step = epoch * len(trainLoader)
        val_step = epoch * len(valLoader)
        minLossEpoch = epoch
        print_log(log_fd, "Resuming from epoch: {}, loss: {}".format(epoch, minLoss))
    print_log(log_fd, "Learning Rate: {}".format(optimizer.param_groups[0]['lr']))
    chamfer = ChamferDistanceL1().to(device)
    # knn = kNNLoss(k=15, n_seeds=50)
    # minLoss = 1e10
    train_step = 0
    model.to(device)
    print("[+] Training Step: {}".format(train_step))
    for epoch in range(epoch + 1, args.epochs + epoch + 1):
        if train_step < 5000:
            x = 0.9
            y = 0.1
            z = 0.0
        elif train_step < 10000:
            x = 0.8
            y = 0.2
            z = 0.0
        elif train_step < 25000:
            x = 0.5
            y = 0.5
            z = 0.0
        else:
            x = 0.2
            y = 0.8
            z = 0.0
        
        model.train()
        train_loss = 0.0
        start = time.time()
        print_log(log_fd, "------------------Epoch: {}------------------".format(epoch))
        for i, (data, labels, names) in enumerate(tqdm(trainLoader)):
            if args.partial:
                gt = data[0]
            else:
                gt = data[1]
            # gt = torch.Tensor(gt).transpose(2, 1).float().to(device)
            gt = gt.to(torch.float32).to(device)
            optimizer.zero_grad()
            coarse, fine = model(gt)
            loss1 = chamfer(coarse, gt)
            loss2 = chamfer(fine, gt)
            # knnloss = knn(fine)
            loss = loss1 * x + loss2 * y
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            loss_log["train"].append(loss.item())
            train_writer.add_scalar('loss', loss.item(), train_step)
            train_writer.add_scalar('coarse', loss1.item(), train_step)
            train_writer.add_scalar('dense', loss2.item(), train_step)
            train_step += 1
            if train_step % min(100, len(trainLoader)//3) == 0:
                recon =  fine.detach().cpu().numpy()
                plot_X = np.stack([np.arange(len(loss_log['train']))], 1)
                plot_Y = np.stack([np.array(loss_log['train'])], 1)
                vis.line(X=plot_X, Y=plot_Y, win=1,
                                    opts={'title': f'PCN Train Loss', 'legend': ['train_loss'], 'xlabel': 'Iteration', 'ylabel': 'Loss'})
                vis.scatter(X=recon[0].reshape(-1, 3), Y=label, win=2,
                                     opts={'title': f"Generated Pointcloud ", 'markersize': 2, 'markercolor': colors, 'webgl': True})
        train_loss /= len(trainLoader)
        end = time.time()
        # print("[+] Epoch: {}, Train Loss: {}, Time: {}".format(epoch, train_loss, end-start))
        lr_scheduler.step()
        print_log(log_fd, "Epoch: {}, Train Loss: {}, Learning_Rate: {}, Time: {}".format(epoch, train_loss, optimizer.param_groups[0]['lr'], end-start))
        
        model.eval()
        val_loss = 0.0
        start = time.time()
        with torch.no_grad():
            rand_iter = random.randint(0, len(valLoader) - 1)
            for i, (data, labels, names) in enumerate(tqdm(valLoader)):
                if args.partial:
                    gt = data[0]
                else:
                    gt = data[1]
                # gt = torch.Tensor(gt).transpose(2, 1).float().to(device)
                gt = gt.to(torch.float32).to(device)
                optimizer.zero_grad()
                coarse, fine = model(gt)
                loss1 = chamfer(coarse, gt)
                loss2 = chamfer(fine, gt)
                # knnloss = knn(fine)
                loss = loss1 * 0.5 + loss2 * 0.5
                val_loss += loss.item()

                if rand_iter == i:
                    index = random.randint(0, fine.shape[0] - 1)
                    plot_pcd_one_view(os.path.join(epochs_dir, 'epoch_{:03d}.png'.format(epoch)),
                                      [coarse[index].detach().cpu().numpy(), fine[index].detach().cpu().numpy(), gt[index].detach().cpu().numpy()],
                                      ['Coarse', 'Dense', 'Ground Truth'], xlim=(-0.5, 1), ylim=(-0.5, 1), zlim=(-0.5, 1))
                    recon =  fine.detach().cpu().numpy()
                    vis.scatter(X=recon[0].reshape(-1, 3), Y=label, win=2,
                                        opts={'title': f"Generated Pointcloud ", 'markersize': 2, 'markercolor': colors, 'webgl': True})

        val_loss /= len(valLoader)
        end = time.time()
        # print("[+] Epoch: {}, Val Loss: {}, Time: {}".format(epoch, val_loss, end-start))
        val_step += 1
        val_writer.add_scalar('ValLoss', val_loss, val_step)
        print_log(log_fd, "Epoch: {}, Val Loss: {}, Time: {}".format(epoch, val_loss, end-start))

        if val_loss < minLoss:
            minLossEpoch = epoch
            minLoss = val_loss
            torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            # "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "epoch": epoch,
            "loss": val_loss,
            "loss_log": loss_log,
            }, bestSavePath)
            # print("[+] Best Model saved")
            print_log(log_fd, "Best Model saved")
        
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            # "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "epoch": epoch,
            "loss": val_loss,
            "loss_log": loss_log,
            }, lastSavePath)
        # print("[+] Last Model saved")
        print_log(log_fd, "Last Model saved")

    print("[+] Best Model saved at epoch: {} with loss {}".format(minLossEpoch, minLoss))
    log_fd.close()

def testModel(model, testLoader, args):
    if not os.path.exists(os.path.join(args.savePath, args.testOut)):
        os.makedirs(os.path.join(args.savePath, args.testOut))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.to(device)
    model.eval()
    test_loss = 0.0
    start = time.time()
    chamfer = ChamferDistanceL1()
    # knn = kNNLoss(k=15, n_seeds=50)
    with torch.no_grad():
        for i, (data, labels, names) in enumerate(tqdm(testLoader)):
            if args.partial:
                gt = data[0]
            else:
                gt = data[1]
            # gt = torch.Tensor(gt).transpose(2, 1).float().to(device)
            gt = gt.to(torch.float32).to(device)
            optimizer.zero_grad()
            coarse, fine = model(gt)
            loss1 = chamfer(coarse, gt)
            loss2 = chamfer(fine, gt)
            # knnloss = knn(fine)
            loss = loss1 * 0.5 + loss2 * 0.50
            test_loss += loss.item()
            if args.testSave:
                save_pcd(fine.squeeze().detach().cpu().numpy(), os.path.join(os.path.join(args.savePath, args.testOut), "{}_fine.pcd".format(names[0])))
                save_pcd(data[0].squeeze(), os.path.join(os.path.join(args.savePath, args.testOut), "{}_partial.pcd".format(names[0])))
                save_pcd(data[1].squeeze(), os.path.join(os.path.join(args.savePath, args.testOut), "{}_gt.pcd".format(names[0])))
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

def load_model_test(model, path):
    print("[+] Loading Model from: {}".format(path))
    checkpoint = torch.load(path)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("[+] Model Statistics - Epoch: {}, Loss: {}".format(checkpoint["epoch"], checkpoint["loss"]))
    except:
        model.load_state_dict(checkpoint)
    return model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="caesar-fitted-meshes-pcd", help="Path to dataset")
    parser.add_argument("--json", type=str, default="data_subset.json", help="Path to json file")
    parser.add_argument("--genFolder", type=str, default="gen", help="Path to generated files")
    parser.add_argument("--testOut", type=str, default="testOut", help="Path to test output")
    parser.add_argument("--savePath", type=str, default=".", help="Path to save model")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--step", type=int, default=5, help="Step size for lr scheduler")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma for lr scheduler")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--modelPath", type=str, default="bestModel.pth", help="Path to model")
    parser.add_argument("--test", action="store_true", help="Test model")
    parser.add_argument("--testSave", action="store_true", help="Save test output")
    parser.add_argument("--partial", action="store_true", help="Use partial point clouds")
    parser.add_argument("--visdom", type=int, default=8097, help="Port for visdom")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()   

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trainLoader, testLoader, valLoader = dataLoaders(args)
    
    model = getModel()

    if args.test:
        model = load_model_test(model, args.modelPath)
        testModel(model, testLoader, args=args)
        exit()

    train(model, trainLoader, valLoader, args=args)
    
    print("[+] Testing Model")
    testModel(model, testLoader, args=args)

    # # Load Model
    # model = load_model(model, bestSavePath)
    
    # print("[+] Testing Model with best model")
    # testModel(model, testLoader, args.testOut, lr=LR, save=testSave, args=args)
    
