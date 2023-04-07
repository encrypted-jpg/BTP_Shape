import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
from tqdm import tqdm
import math


def o3d_visualize_pc(pc):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([point_cloud])

def plot_pcd_one_view_reg(filename, pcds, titles, suptitle='', sizes=None, cmap='Reds', zdir='y',
                         xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5)):
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 4, 4))
    # elev = 90
    # azim = -90
    for j, (pcd, size) in enumerate(zip(pcds, sizes)):
        if j == 3:
            ax = fig.add_subplot(1, len(pcds), j + 1, projection='3d')
            ax.view_init(elev, azim)
            ax.scatter(pcds[0][:, 0], pcds[0][:, 1], pcds[0][:, 2], zdir=zdir, c=pcds[0][:, 0], s=size, cmap=cmap, vmin=-1.0, vmax=1.0)
            ax.scatter(pcds[2][:, 0], pcds[2][:, 1], pcds[2][:, 2], zdir=zdir, c=pcds[2][:, 0], s=size, cmap='magma', vmin=-1.0, vmax=1.0)
            ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
        else:
            color = pcd[:, 0]
            ax = fig.add_subplot(1, len(pcds), j + 1, projection='3d')
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1.0, vmax=1.0)
            ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)

def plot_pcd_one_view(filename, pcds, titles, suptitle='', sizes=None, cmap='Reds', zdir='y',
                         xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5)):
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 4, 4))
    # elev = 90
    # azim = -90
    for j, (pcd, size) in enumerate(zip(pcds, sizes)):
        color = pcd[:, 0]
        ax = fig.add_subplot(1, len(pcds), j + 1, projection='3d')
        ax.view_init(elev, azim)
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1.0, vmax=1.0)
        ax.set_title(titles[j])
        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)

def plot_all(filename, pcds, files, suptitle='', sizes=None, cmap='Reds', zdir='y',
                         xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5)):
    n = 4
    if len(pcds) > n:
        for i in tqdm(range(0, len(pcds), n)):
            plot_pcd_one_view_reg(filename.replace(".png", "") + "_" + str(i) + ".png", pcds[i:i+n], files[i:i+n], suptitle, sizes, cmap, zdir, xlim, ylim, zlim)
    else:
        plot_pcd_one_view(filename, pcds, files, suptitle, sizes, cmap, zdir, xlim, ylim, zlim)

def rotate_caesar(points, angle=120):
    angle = angle * math.pi / 180
    R = np.array([[math.cos(angle), -math.sin(angle), 0],
                  [math.sin(angle), math.cos(angle), 0],
                  [0, 0, 1]])
    return np.dot(points, R)

def rotate_dfaust(points, angle=120):
    angle = angle * math.pi / 180
    R = np.array([[math.cos(angle), 0, math.sin(angle)],
                  [0, 1, 0],
                  [-math.sin(angle), 0, math.cos(angle)]])
    return np.dot(points, R)

# 0 -> Caesar, 1 -> Dfaust
dataset = 1
if dataset == 0:
    elev = 90
    azim = -90
    l = 0.0
    r = 1
    size = 1
    rotate = rotate_caesar
elif dataset == 1:
    elev = 0
    azim = -120
    l = -0.5
    r = 0.7
    size = 2
    rotate = rotate_dfaust

xlim = (l, r)
ylim = (l, r)
zlim = (l, r)

# for file in tqdm(os.listdir("temp")):
#     if file.endswith(".pcd"):
#         pcd = o3d.io.read_point_cloud("temp/" + file)
#         pcd = np.asarray(pcd.points)
#         file = file.replace(".pcd", "")
#         plot_pcd_one_view("temp/" + file + ".png", [pcd], [file], 
#                           suptitle="", sizes=[1], cmap='Reds', zdir='y',
#                             xlim=xlim, ylim=ylim, zlim=zlim)

pcds = []
files = []
for file in tqdm(os.listdir("temp")):
    if file.endswith(".pcd"):
        pcd = o3d.io.read_point_cloud("temp/" + file)
        pcd = np.asarray(pcd.points)
        pcds.append(rotate(pcd, angle=120))
        # pcds.append(pcd)
        files.append(file.replace(".pcd", ""))

files = ["Partial", "Nearest", "Nearest_Down-Sampled", "Registered"] * (len(files)//4)

plot_all("temp/" + "all.png", pcds, files,
                    suptitle="", sizes=[1 for i in range(len(pcds))], cmap='viridis', zdir='y',
                    xlim=xlim, ylim=ylim, zlim=zlim)

