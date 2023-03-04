import os
import open3d as o3d
import numpy as np
import torch
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from extensions.chamfer_dist import ChamferDistanceL1
import time


# def chamferDistanceL1(points1, points2):
#     """
#     points1: (N, 3)
#     points2: (M, 3)
#     Compute L1 Chamfer Distance between to point clouds of different shapes using GPU
#     """
#     points1 = torch.from_numpy(points1).float().cuda()
#     points2 = torch.from_numpy(points2).float().cuda()
#     dist1 = torch.cdist(points1, points2, p=1)
#     dist2 = torch.cdist(points2, points1, p=1)
#     return torch.mean(dist1.min(dim=1)[0]) + torch.mean(dist2.min(dim=1)[0])

def loadPCD(path):
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    return points

def loadCompletePCDFromDir(folder):
    """
    Return a list of complete point clouds along with the file name without extension
    """
    complete = []
    for f in tqdm(os.listdir(folder)):
        if f.endswith(".pcd"):
            complete.append((loadPCD(os.path.join(folder, f))[::3], os.path.basename(f).replace(".pcd", "")))
    return complete

def loadPartialPCDFromDir(folder):
    """
    Return a list of partial point clouds along with the file name without extension.
    With following folder structure:
    |-- partial
        |-- *
            |-- *-[1-8].pcd
    """
    partial = []
    for f in tqdm(os.listdir(folder)):
        if os.path.isdir(os.path.join(folder, f)):
            for s in os.listdir(os.path.join(folder, f)):
                if s.endswith(".pcd"):
                    partial.append((loadPCD(os.path.join(folder, f, s)), os.path.basename(s).split("-")[0]))
    return partial

def similarComplete(partial, complete, chamferDist):
    """
    Find the complete point cloud that is similar to the partial point cloud using chamfer distance.
    Print the most similar complete point clouds with their distance.
    """
    partial, partialName = partial
    complete, completeNames = zip(*complete)
    dist = 1000000.0
    name = None
    for i, c in enumerate(complete):
        d = chamferDist(partial, c).item()
        if d < dist:
            dist = d
            name = completeNames[i]
    print("Most similar complete point cloud for partial point cloud: {}".format(partialName))
    print("Distance: {:.4f}, Name: {}".format(dist, name))

def similarComplete10(partial, complete, chamferDist):
    """
    Find the complete point cloud that is similar to the partial point cloud using chamfer distance.
    Print the top-10 similar complete point clouds with their distance.
    """
    partial, partialName = partial
    complete, completeNames = zip(*complete)
    distances = []
    for c in complete:
        distances.append(chamferDist(partial, c).item())
    distances = np.array(distances)
    indices = np.argsort(distances)
    print("Top-10 similar complete point clouds for partial point cloud: {}".format(partialName))
    for i in range(10):
        print("Distance: {:.4f}, Name: {}".format(distances[indices[i]], completeNames[indices[i]]))

if __name__ == "__main__":
    partial = loadPartialPCDFromDir("caesar-fitted-meshes-pcd/test/partial")
    complete = loadCompletePCDFromDir("caesar-fitted-meshes-pcd/train/complete")
    # Convert partial and complete to numpy array and then to tensor array and then to cuda tensor
    partial = [(torch.from_numpy(p).float().cuda().unsqueeze(0), n) for p, n in partial]
    complete = [(torch.from_numpy(c).float().cuda().unsqueeze(0), n) for c, n in complete]
    print(partial[0][0].shape, complete[0][0].shape)
    chamferDist = ChamferDistanceL1()

    start = time.time()
    for i in range(50):
        similarComplete(partial[i], complete, chamferDist)
    end = time.time()
    print("Time taken for 1 similarity : {:.4f}".format(end - start))

    start = time.time()
    for i in range(50):
        similarComplete10(partial[i], complete, chamferDist)
    end = time.time()
    print("Time taken for 10 similarity : {:.4f}".format(end - start))
