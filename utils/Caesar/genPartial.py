import os
import open3d as o3d
import numpy as np
from tqdm import tqdm
import torch

def genPartialWithKCenters(pcdfile, num_centers=15, k=100, partialfile=None):
    """
    This function initially reduces the density of the point cloud by half
    and then selects num_centers points from the point cloud and then removes the k-nearest points.
    Then randomly select 1000 points from the remaining points.
    Finally save the point cloud as a partial point cloud.
    """
    pcd = o3d.io.read_point_cloud(pcdfile)
    if not partialfile:
        partialfile = pcdfile.replace(".pcd", "-partial.pcd")
    points = np.asarray(pcd.points)
    points = points[::2, :]
    centers = points[np.random.choice(points.shape[0], num_centers, replace=False), :]
    # Create a new point cloud object and copy the points to it
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    tree = o3d.geometry.KDTreeFlann(pcd)
    [k, idx, _] = tree.search_knn_vector_3d(centers[0], k)
    num_centers = min(num_centers, points.shape[0]//k)
    for i in range(1, num_centers):
        [k, idx2, _] = tree.search_knn_vector_3d(centers[i], k)
        idx = np.concatenate((idx, idx2))
    idx = np.unique(idx)
    points = np.delete(points, idx, axis=0)
    points = points[np.random.choice(points.shape[0], 1024, replace=False), :]
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(partialfile, pcd)

def folderGenPartial(folder):
    print("Generating partial point clouds in folder: ", folder)
    y = os.path.join(folder, "complete")
    x = os.path.join(folder, "partial")

    if not os.path.exists(x):
        os.mkdir(x)

    for f in tqdm(os.listdir(y)):
        seeds = [1, 2, 3, 4, 5, 6, 7, 8]
        for seed in seeds:
            np.random.seed(seed)
            partialFolder = os.path.join(x, f.replace(".pcd", ""))
            if not os.path.exists(partialFolder):
                os.mkdir(partialFolder)
            genPartialWithKCenters(os.path.join(y, f), partialfile=os.path.join(partialFolder, f.replace(".pcd", f"-{seed}.pcd")))



if __name__ == "__main__":
    folder = "caesar-fitted-meshes-pcd"
    folderGenPartial(folder)