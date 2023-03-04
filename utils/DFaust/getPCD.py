import os
import open3d as o3d
import numpy as np
import json
import h5py
from tqdm import tqdm

def save_pcd(points, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)

def genPcds(filePath):
    print("Generating point clouds in folder: ", filePath)
    file = h5py.File(filePath, "r")
    
    path = "D-Faust"
    fpath = filePath.split(".")[0]
    savePath = os.path.join(path, fpath, "complete")
    os.makedirs(savePath, exist_ok=True)

    for x, y in file.items():
        if x == "faces":
            continue
        verts = y[()].transpose([2, 0, 1])
        cpath = os.path.join(savePath, x)
        os.makedirs(cpath, exist_ok=True)
        for i in tqdm(range(verts.shape[0])):
            save_pcd(verts[i], os.path.join(cpath, str(i)+".pcd"))
        print(x, verts.shape)


if __name__ == "__main__":
    filePath = "registrations_f.hdf5"
    genPcds(filePath)