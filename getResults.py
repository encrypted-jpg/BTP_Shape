from models import *
import os


def print_log(fd,  message, time=True):
    if time:
        message = ' ==> ' + message
    fd.write(message + '\n')
    fd.flush()
    print(message)

def load_model(model, path, fd):
    if path == None:
        return None
    print_log(fd, "[+] Loading Model from: {}".format(path))
    checkpoint = torch.load(path)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        print_log(fd, "[+] Model Statistics - Epoch: {}, Loss: {}".format(checkpoint["epoch"], 10000 * checkpoint["loss"]))
    except:
        model.load_state_dict(checkpoint)


lst = ["caesar", "dfaust"]
tps = ["GTvsGT", "PvsGT", "RegvsGT", "RegvsGTDCD"]
model = PCN(6144, 1024, 4)
fd = open("results.txt", "w")
for name in lst:
    for tp in tps:
        folder = name+tp
        for x in os.listdir(folder):
            if x.startswith("bestModel"):
                print_log(fd, "Dataset: {}, Type: {}".format(name, tp))
                load_model(model, os.path.join(folder, x), fd)
    print_log(fd, "=============================")
fd.close()
