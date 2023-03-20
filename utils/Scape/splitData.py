import os
from shutil import move
from tqdm import tqdm
import numpy as np
import json

def split_train_test_val(folder, train=0.8, test=0.1, val=0.1):
    """
    Get the list of files in the complete folder.
    Then Randomize it and split it into train, test and val with indexing.
    Print the counts of each at the end.
    Current Folder structure:
    caesar-fitted-meshes-pcd
    |-- complete
        |-- *.pcd
    |-- partial
        |-- *
            |-- *-[1-8].pcd
    The folder structure should be as follows:
    caesar-fitted-meshes-pcd
    |-- train
    |   |-- complete
    |   |-- partial
    |-- test
    |   |-- complete
    |   |-- partial
    |-- val
        |-- complete
        |-- partial
    """
    print("Splitting data into train, test and val...")
    if not os.path.exists(os.path.join(folder, "train")):
        os.mkdir(os.path.join(folder, "train"))
        os.mkdir(os.path.join(folder, "train", "complete"))
        os.mkdir(os.path.join(folder, "train", "partial"))
    if not os.path.exists(os.path.join(folder, "test")):
        os.mkdir(os.path.join(folder, "test"))
        os.mkdir(os.path.join(folder, "test", "complete"))
        os.mkdir(os.path.join(folder, "test", "partial"))
    if not os.path.exists(os.path.join(folder, "val")):
        os.mkdir(os.path.join(folder, "val"))
        os.mkdir(os.path.join(folder, "val", "complete"))
        os.mkdir(os.path.join(folder, "val", "partial"))
    files = os.listdir(os.path.join(folder, "complete"))
    np.random.shuffle(files)
    train_index = int(len(files) * train)
    test_index = int(len(files) * test)
    train_files = files[:train_index]
    test_files = files[train_index:train_index+test_index]
    val_files = files[train_index+test_index:]
    for f in tqdm(train_files):
        move(os.path.join(folder, "complete", f), os.path.join(folder, "train", "complete", f))
        move(os.path.join(folder, "partial", f.replace(".pcd", "")), os.path.join(folder, "train", "partial", f.replace(".pcd", "")))
    for f in tqdm(test_files):
        move(os.path.join(folder, "complete", f), os.path.join(folder, "test", "complete", f))
        move(os.path.join(folder, "partial", f.replace(".pcd", "")), os.path.join(folder, "test", "partial", f.replace(".pcd", "")))
    for f in tqdm(val_files):
        move(os.path.join(folder, "complete", f), os.path.join(folder, "val", "complete", f))
        move(os.path.join(folder, "partial", f.replace(".pcd", "")), os.path.join(folder, "val", "partial", f.replace(".pcd", "")))
    print("Train: ", len(train_files))
    print("Test: ", len(test_files))
    print("Val: ", len(val_files))

def createJSON(folder):
    """
    This function creates a single JSON file with the following structure:
    |-- train
    |   |-- *
    |-- test
    |   |-- *
    |-- val
        |-- *
    File extension is not included in the list.
    At the end prettify the json file.
    """
    print("Creating JSON file...")
    data = {}
    data["train"] = []
    data["test"] = []
    data["val"] = []
    for f in tqdm(os.listdir(os.path.join(folder, "train", "complete"))):
        data["train"].append(f.replace(".pcd", ""))
    for f in tqdm(os.listdir(os.path.join(folder, "test", "complete"))):
        data["test"].append(f.replace(".pcd", ""))
    for f in tqdm(os.listdir(os.path.join(folder, "val", "complete"))):
        data["val"].append(f.replace(".pcd", ""))
    with open(os.path.join(folder, "data.json"), "w") as f:
        json.dump(data, f, indent=4)

def folderSplitJSON(folder):
    split_train_test_val(folder)
    createJSON(folder)


if __name__ == "__main__":
    folder = "scape-pcd"
    folderSplitJSON(folder)
