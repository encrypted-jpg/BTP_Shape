import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import open3d as o3d
import argparse
import json
from models import *
from datasets.caesarDataset import CaesarDataset
from datasets.dfaustDataset import DFaustDataset
from extensions.chamfer_dist import ChamferDistanceL1
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
import datetime
from utils.visual import plot_pcd_one_view
import random


def print_log(fd,  message, time=True):
    if time:
        message = ' ==> '.join([datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), message])
    fd.write(message + '\n')
    fd.flush()
    print(message)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot',  default=".", help='path to dataset')
    parser.add_argument('--json', default="data.json", help='path to json file')
    parser.add_argument('--batchSize', type=int, default=24, help='input batch size')
    parser.add_argument('--num_dense', type=int, default=6144, help='the point number of a sample')
    parser.add_argument('--num_partial', type=int, default=384, help='the point number of a sample')
    parser.add_argument('--partial', action='store_true', help='use partial point cloud as input')
    parser.add_argument('--niter', type=int, default=201, help='number of epochs to train for')
    parser.add_argument('--latent_dim', type=int, default=1024, help='latent dimension')
    parser.add_argument('--grid_size', type=int, default=4, help='grid size')
    parser.add_argument('--num_dense', type=int, default=6144, help='number of dense points')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--model', type=str, default = '',  help='model path')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--save_dir', default = 'checkpoints', help = 'save directory')
    parser.add_argument('--img_freq', type=int, default=100, help='frequency of saving images')
    opt = parser.parse_args()
    return opt

def get_models(opt):
    model = PCNAE(latent_dim=opt.latent_dim, grid_size=opt.grid_size, num_dense=opt.num_dense)
    return model


def get_dataLoaders(f, opt):
    print_log(f, "Loading the data...")
    folder = opt.dataroot
    json = opt.json
    batch_size = opt.batchSize
    gt_num_points = opt.num_dense
    if "caesar" in folder.lower():
        trainDataset = CaesarDataset(folder, json, partition="train", seeds=1, gt_num_points=gt_num_points)
        testDataset = CaesarDataset(folder, json, partition="test", seeds=1, gt_num_points=gt_num_points)
        valDataset = CaesarDataset(folder, json, partition="val", seeds=1, gt_num_points=gt_num_points)
    elif "dfaust" in folder.lower():
        trainDataset = DFaustDataset(folder, json, partition="train", seeds=1, gt_num_points=gt_num_points)
        testDataset = DFaustDataset(folder, json, partition="test", seeds=1, gt_num_points=gt_num_points)
        valDataset = DFaustDataset(folder, json, partition="val", seeds=1, gt_num_points=gt_num_points)
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=1, shuffle=True)
    valLoader = DataLoader(valDataset, batch_size=batch_size, shuffle=True)
    return trainLoader, testLoader, valLoader

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_pcd(points, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Conv1d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0) 

def load_model(model, path, f):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    resume_epoch = checkpoint['epoch']
    if 'loss' in checkpoint.keys():
        loss = checkpoint['loss']
    else:
        loss = None
    print_log(f, "Loaded {} Model with Epoch {} Loss {}".format(path, resume_epoch, loss))
    return model

def pre_ops(opt):
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    # print_log(f, "Random Seed: " + str(opt.manualSeed))
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    if not os.path.exists(os.path.join(opt.save_dir, 'pcds')):
        os.makedirs(os.path.join(opt.save_dir, 'pcds'))
    if not os.path.exists(os.path.join(opt.save_dir, 'imgs')):
        os.makedirs(os.path.join(opt.save_dir, 'imgs'))
   
def train(model, trainLoader, testLoader, opt):
    chamfer_loss = ChamferDistanceL1().to(device)

    f=open(os.path.join(opt.save_dir, 'loss_PFNet.txt'),'a')

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_loss = 1000000
    for epoch in range(0, opt.niter):
        if epoch<15:
            fwt = 0.1
        elif epoch<30:
            fwt = 0.5
        else:
            fwt = 1.0
        
        train_loss = train_epoch(model, trainLoader, epoch, fwt, chamfer_loss, optimizer, opt)
        
        test_loss = test_epoch(model, testLoader, epoch, chamfer_loss, opt)

        scheduler.step()
        print_log(f, 'Learning Rate: {}'.format(scheduler.get_last_lr()))
        if opt.partial:
            mstr = "partial"
        else:
            mstr = "complete"
        torch.save({'epoch':epoch+1,
                    'loss': test_loss,
                    'state_dict':model.state_dict()},
                    os.path.join(opt.save_dir, 'pcn_{}.pth'.format(mstr) ))
        print_log(f, 'Saved Last Model')

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({'epoch':epoch+1,
                        'loss': test_loss,
                        'state_dict':model.state_dict()},
                        os.path.join(opt.save_dir, 'pcn_{}_best.pth'.format(mstr) ))
            print_log(f, 'Saved Best Model')

def train_epoch(model, trainLoader, epoch, fwt, chamfer_loss, optimizer, opt):

    f=open(os.path.join(opt.save_dir, 'loss_PCRPCN.txt'),'a')

    model.train()
    
    train_loss = 0.0
    start = time.time()
    for i, data in enumerate(trainLoader, 0):
        
        (partial, gt), _, _ = data
        pc = gt
        if opt.partial:
            pc = partial
        # partial = partial.to(device)
        # gt = gt.to(device)
        pc = pc.to(device)
        # Train Model
        optimizer.zero_grad()
        tc, tpc, coarse, fine = model(pc)
        coarse_loss = chamfer_loss(coarse, pc)
        fine_loss = chamfer_loss(fine, pc)

        loss = coarse_loss + fwt * fine_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()


        if i % opt.img_freq == 0:
            print_log(f, 'Train Epoch: {} [{}/{} ({:.0f}%) - {}]\tLoss: {:.6f}\tCoarse Loss: {:.6f}\tFine Loss: {:.6f}\tTime: {:.4f}'.format(
                epoch, i * len(partial), len(trainLoader.dataset),
                100. * i / len(trainLoader), i, loss.item(), coarse_loss.item(), fine_loss.item(), time.time() - start))
            # if not os.path.exists(os.path.join(opt.save_dir, 'pcds')):
            #     os.makedirs(os.path.join(opt.save_dir, 'pcds'))
            if not os.path.exists(os.path.join(opt.save_dir, 'imgs')):
                os.makedirs(os.path.join(opt.save_dir, 'imgs'))
            # save_pcd(real_center[0].cpu().detach().numpy().reshape(-1, 3), '{}/train_{}_{}_real.pcd'.format(os.path.join(opt.save_dir, 'pcds'), epoch, i))
            # save_pcd(fake[0].cpu().detach().numpy().reshape(-1, 3), '{}/train_{}_{}_fake.pcd'.format(os.path.join(opt.save_dir, 'temp'), epoch, i))
            plot_pcd_one_view(os.path.join(opt.save_dir, 'imgs', 'train_{}_{}.png'.format(epoch, i)),
                                [gt[0].cpu().detach().numpy().reshape(-1, 3), coarse[0].cpu().detach().numpy().reshape(-1, 3), fine[0].cpu().detach().numpy().reshape(-1, 3)],
                                ['Ground Truth', 'Coarse', 'Fine'], xlim=[-1, 1], ylim=[-1, 1], zlim=[-1, 1])
            # plot_pcd_one_view(os.path.join(opt.save_dir, 'imgs', 'train_{}_{}_temp.png'.format(epoch, i)),
            #                   [tc[0].cpu().detach().numpy().reshape(-1, 3), tpc[0].cpu().detach().numpy().reshape(-1, 3), partial[0].cpu().detach().numpy().reshape(-1, 3)],
            #                   ['Template', 'Transformed', 'Source'], xlim=[-1, 1], ylim=[-1, 1], zlim=[-1, 1])

    train_loss /= len(trainLoader)
    print_log(f, 'Epoch: {} Average loss: {:.4f}, Time: {:.4f}'.format(epoch, train_loss, time.time() - start))
    f.close()
    return train_loss

def test_epoch(model, testLoader, epoch, fwt, chamfer_loss):
    f = open(os.path.join(opt.save_dir, 'loss_PCRPCN.txt'),'a')
    model.eval()
    start = time.time()
    test_loss = 0.0
    with torch.no_grad():
        rid = random.randint(0, len(testLoader)-1)
        for i, data in enumerate(testLoader, 0):
            
            (partial, gt), _, _ = data
            partial = partial.to(device)
            gt = gt.to(device)

            # Train Model
            sf, tf, coarse, fine = model(partial)
            coarse_loss = chamfer_loss(coarse, gt)
            fine_loss = chamfer_loss(fine, gt)

            loss = coarse_loss + fwt * fine_loss
            
            test_loss += loss.item()

            if i == rid:
                print_log(f, 'Test Epoch: {} [{}/{} ({:.0f}%) - {}]\tLoss: {:.6f}\tCoarse Loss: {:.6f}\tFine Loss: {:.6f}\tTime: {:.4f}'.format(
                    epoch, i * len(partial), len(testLoader.dataset),
                    100. * i / len(trainLoader), i, loss.item(), coarse_loss.item(), fine_loss.item(), time.time() - start))
                # if not os.path.exists(os.path.join(opt.save_dir, 'pcds')):
                #     os.makedirs(os.path.join(opt.save_dir, 'pcds'))
                if not os.path.exists(os.path.join(opt.save_dir, 'imgs')):
                    os.makedirs(os.path.join(opt.save_dir, 'imgs'))
                # save_pcd(real_center[0].cpu().detach().numpy().reshape(-1, 3), '{}/train_{}_{}_real.pcd'.format(os.path.join(opt.save_dir, 'pcds'), epoch, i))
                # save_pcd(fake[0].cpu().detach().numpy().reshape(-1, 3), '{}/train_{}_{}_fake.pcd'.format(os.path.join(opt.save_dir, 'temp'), epoch, i))
                plot_pcd_one_view(os.path.join(opt.save_dir, 'imgs', 'test_{}_{}.png'.format(epoch, i)),
                                    [gt[0].cpu().detach().numpy().reshape(-1, 3), coarse[0].cpu().detach().numpy().reshape(-1, 3), fine[0].cpu().detach().numpy().reshape(-1, 3)],
                                    ['Ground Truth', 'Coarse', 'Fine'], xlim=[-1, 1], ylim=[-1, 1], zlim=[-1, 1])
    
    end = time.time()
    test_loss /= len(testLoader)
    print_log(f, 'Epoch: %d, test loss: %.4f, Time: %.4f' % (epoch, test_loss,end-start))
    
    f.close()
    return test_loss
if __name__ == "__main__":
    opt = get_parser()
    pre_ops(opt)
    f = open(os.path.join(opt.save_dir, 'loss_PFNet.txt'),'w')
    print_log(f, str(opt))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainLoader, testLoader, trainDataset, testDataset = get_dataLoaders(f, opt)

    # start = time.time()
    # for i, data in tqdm(enumerate(trainLoader, 0)):
    #     pass
    # end = time.time()
    # print_log(f, 'Time for loading data: %.4f' % (end-start))
    
    model = get_models(opt)

    if torch.cuda.is_available():
        print_log(f, "Using GPU")
        model = model.cuda()

    if opt.model != '':
        load_model(model, opt.model)
    
    print_log(f, "Total Number of Parameters in Model: {:.3f}M".format(count_parameters(model)/1e6))

    f.close()

    train(model, trainLoader, testLoader, opt)



