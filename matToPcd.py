import scipy.io.matlab as mio
import open3d as o3d
import os
from tqdm import tqdm

def matToPcd(matfile, pcdfile=None):
    mat = mio.loadmat(matfile)
    if not pcdfile:
        pcdfile = matfile.replace(".mat", ".pcd")
    
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        try:
            if 1 in v.shape:
                v=v.flatten()
            # Normalize the numpy array v to be between 0 and 1
            v = (v - v.min()) / (v.max() - v.min())
            # print(v.shape, type(v))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(v)
            o3d.io.write_point_cloud(pcdfile, pcd)
        except:
            print("ERROR: ", k, v)

def folderMatToPcd(folder):
    print("Converting .mat files to .pcd files in folder: ", folder)
    pcdFolder = f"{folder}-pcd"

    if not os.path.exists(pcdFolder):
        os.mkdir(pcdFolder)
    
    if not os.path.exists(os.path.join(pcdFolder, "complete")):
        os.mkdir(os.path.join(pcdFolder, "complete"))

    for f in tqdm(os.listdir(folder)):
        if f.endswith(".mat"):
            matToPcd(os.path.join(folder, f), os.path.join(pcdFolder, "complete", f.replace(".mat", ".pcd")))
    return pcdFolder

if __name__ == "__main__":
    folder = "caesar-fitted-meshes"
    
    folderMatToPcd(folder)

