import open3d as o3d
import os
from tqdm import tqdm

def plyToPcd(plyFile, pcdFile):
    pcd = o3d.io.read_point_cloud(plyFile)
    pcd = pcd.voxel_down_sample(voxel_size=0.0185)
    o3d.io.write_point_cloud(pcdFile, pcd)

def folderPlyToPcd(folder):
    print("Converting .ply files to .pcd files in folder: ", folder)
    pcdFolder = f"{folder}-pcd"

    if not os.path.exists(pcdFolder):
        os.mkdir(pcdFolder)
    
    if not os.path.exists(os.path.join(pcdFolder, "complete")):
        os.mkdir(os.path.join(pcdFolder, "complete"))

    for f in tqdm(os.listdir(folder)):
        if f.endswith(".ply"):
            plyToPcd(os.path.join(folder, f), os.path.join(pcdFolder, "complete", f.replace(".ply", ".pcd")))
    return pcdFolder

if __name__ == "__main__":
    folder = "scape"
    
    folderPlyToPcd(folder)

