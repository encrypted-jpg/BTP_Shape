from getPCD import genPcds
from genPartial import folderGenPartial
from splitData import split_train_test_val
import os


def genData(filePath):
    folder = os.path.join("D-Faust", filePath.split(".")[0])
    genPcds(filePath)
    folderGenPartial(folder)
    split_train_test_val(folder)

if __name__ == "__main__":
    genData("registrations_f.hdf5")
    genData("registrations_m.hdf5")
