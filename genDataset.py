from matToPcd import folderMatToPcd
from genPartial import folderGenPartial
from splitData import folderSplitJSON

folder = "caesar-fitted-meshes"
# pcdFolder = folderMatToPcd(folder)
# folderGenPartial(pcdFolder)
pcdFolder = folder+"-pcd"
folderSplitJSON(pcdFolder)