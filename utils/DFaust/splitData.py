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
    registration_f
    |-- complete
        |-- *
            |-- *.pcd
    |-- partial
        |-- *
            |-- *.pcd
    The folder structure should be as follows:
    registration_f
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
    for cat in tqdm(os.listdir(os.path.join(folder, "complete"))):
        files = os.listdir(os.path.join(folder, "complete", cat))
        os.mkdir(os.path.join(folder, "train", "complete", cat))
        os.mkdir(os.path.join(folder, "train", "partial", cat))
        os.mkdir(os.path.join(folder, "test", "complete", cat))
        os.mkdir(os.path.join(folder, "test", "partial", cat))
        os.mkdir(os.path.join(folder, "val", "complete", cat))
        os.mkdir(os.path.join(folder, "val", "partial", cat))
        np.random.shuffle(files)
        train_index = int(len(files) * train)
        test_index = int(len(files) * test)
        train_files = files[:train_index]
        test_files = files[train_index:train_index+test_index]
        val_files = files[train_index+test_index:]
        for f in train_files:
            move(os.path.join(folder, "complete", cat, f), os.path.join(folder, "train", "complete", cat, f))
            move(os.path.join(folder, "partial", cat, f), os.path.join(folder, "train", "partial", cat, f))
        for f in test_files:
            move(os.path.join(folder, "complete", cat, f), os.path.join(folder, "test", "complete", cat, f))
            move(os.path.join(folder, "partial", cat, f), os.path.join(folder, "test", "partial", cat, f))
        for f in val_files:
            move(os.path.join(folder, "complete", cat, f), os.path.join(folder, "val", "complete", cat, f))
            move(os.path.join(folder, "partial", cat, f), os.path.join(folder, "val", "partial", cat, f))


def createAllJSON():
    """
    Create a json file with the list of files in the train, test and val folders.
    """
    print("Creating json file...")
    mfolder = "D-Faust/"
    jsonFile = os.path.join(mfolder, "data.json")
    data = {}
    data["train"] = []
    data["test"] = []
    data["val"] = []
    for folder in ["registrations_f", "registrations_m"]:
        folder = os.path.join(mfolder, folder)
        for cat in tqdm(os.listdir(os.path.join(folder, "train", "complete"))):
            files = os.listdir(os.path.join(folder, "train", "complete", cat))
            for f in files:
                data["train"].append(os.path.join(folder, "train", "*", cat, f).replace("\\", "/"))
        for cat in tqdm(os.listdir(os.path.join(folder, "test", "complete"))):
            files = os.listdir(os.path.join(folder, "test", "complete", cat))
            for f in files:
                data["test"].append(os.path.join(folder, "test", "*", cat, f).replace("\\", "/"))
        for cat in tqdm(os.listdir(os.path.join(folder, "val", "complete"))):
            files = os.listdir(os.path.join(folder, "val", "complete", cat))
            for f in files:
                data["val"].append(os.path.join(folder, "val", "*", cat, f).replace("\\", "/"))
    with open(jsonFile, "w") as f:
        json.dump(data, f, indent=4)

def createSepJSON(folder):
    print("Creating json file...")
    mfolder = "D-Faust/"
    jsonFile = os.path.join(mfolder, f"{folder}.json")
    data = {}
    data["train"] = []
    data["test"] = []
    data["val"] = []
    folder = os.path.join(mfolder, folder)
    for cat in tqdm(os.listdir(os.path.join(folder, "train", "complete"))):
        files = os.listdir(os.path.join(folder, "train", "complete", cat))
        for f in files:
            data["train"].append(os.path.join(folder, "train", "*", cat, f).replace("\\", "/"))
    for cat in tqdm(os.listdir(os.path.join(folder, "test", "complete"))):
        files = os.listdir(os.path.join(folder, "test", "complete", cat))
        for f in files:
            data["test"].append(os.path.join(folder, "test", "*", cat, f).replace("\\", "/"))
    for cat in tqdm(os.listdir(os.path.join(folder, "val", "complete"))):
        files = os.listdir(os.path.join(folder, "val", "complete", cat))
        for f in files:
            data["val"].append(os.path.join(folder, "val", "*", cat, f).replace("\\", "/"))
    with open(jsonFile, "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    folder = "D-Faust/registrations_m/"
    # split_train_test_val(folder)
    createAllJSON()
    createSepJSON("registrations_f")
    createSepJSON("registrations_m")