import sol1
from util import *
from validator.Validate import DoWork
import warnings
warnings.filterwarnings( "ignore", module = "vrpy\..*" )
import os
import sys

algorithms = {'sol1': sol1}
instanceNr = [i for i in range(1,21)]

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class args:
    def __init__(self, instancepath, solutionpath):
        self.instance = instancepath
        self.solution = solutionpath
        self.type = 'txt'
        self.itype = 'txt'


def solveBatch(algo, dir):
    print(dir.strip("./") in os.listdir())
    if dir.strip("./") in os.listdir():
        print("Warning:: Directory already exists. Overwrite?")
        if input('[y/n] :') == 'n':
           return
    else: os.mkdir(dir)
    for i in instanceNr:
        print(i)
        instance = loadInstance(i)
        blockPrint()
        savedPath = algo.solveAndSave(instance, dir, i)
        enablePrint()
        DoWork(args(instancePath(i), savedPath))
        
if __name__ == "__main__":
    print("Solving for ", instanceNr)
    algo = algorithms['sol1']
    dir = "./alg1_batch"
    solveBatch(algo, dir)



