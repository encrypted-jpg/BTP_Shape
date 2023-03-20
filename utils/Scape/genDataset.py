from plyToPcd import folderPlyToPcd
from genPartial import folderGenPartial
from splitData import folderSplitJSON

folder = "scape"
pcdFolder = folderPlyToPcd(folder)
folderGenPartial(pcdFolder, n_seeds=16)
pcdFolder = folder+"-pcd"
folderSplitJSON(pcdFolder)