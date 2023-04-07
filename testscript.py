import os
import subprocess
import sys


def get_results():
    cmd1 = "python TrainP.py --folder D-Faust --modelPath dfaustPvsGT/bestModel33.pth --test --partial "

    for file in os.listdir("D-Faust"):
        if file.endswith(".json"):
            file = os.path.basename(file)
            cmd = cmd1 + "--json {}".format(file)
            print(cmd)
            # write output to a file
            with open("dfaustPvsGT/{}.txt".format(file), "w") as f:
                subprocess.call(cmd.split(), stdout=f)

    cmd2 = "python TrainPC.py --folder D-Faust-Reg --modelPath dfaustRegvsGT/bestModel.pth --test --partial "

    for file in os.listdir("D-Faust"):
        if file.endswith(".json"):
            file = os.path.basename(file)
            cmd = cmd2 + "--json {}".format(file)
            print(cmd)
            # write output to a file
            with open("dfaustRegvsGT/{}.txt".format(file), "w") as f:
                subprocess.call(cmd.split(), stdout=f)

    cmd3 = "python TrainPC.py --folder D-Faust-Reg --modelPath dfaustRegvsGTDCD/bestModel.pth --test --partial "

    for file in os.listdir("D-Faust"):
        if file.endswith(".json"):
            file = os.path.basename(file)
            cmd = cmd3 + "--json {}".format(file)
            print(cmd)
            # write output to a file
            with open("dfaustRegvsGTDCD/{}.txt".format(file), "w") as f:
                subprocess.call(cmd.split(), stdout=f)

