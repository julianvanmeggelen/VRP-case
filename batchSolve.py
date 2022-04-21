import importlib.util
import argparse
import sys
import os
from util import *
from validator.Validate import DoWork
import warnings
import logging
warnings.filterwarnings("ignore", module="vrpy\..*")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.CRITICAL)

##
#   USAGE
#   python batchSolve.py
#       -a: algo name must be in algorithms dict below.
#           Algo name is name of module, which must have method solveAndSave whith signature (instance: InstanceCO22, path: Str, i: int) -> Str (path of saved file)
#       -i: up untill instance
#       -d: savedirectory
##

algorithms = ['algorithm1_1', 'algorithm1_2', 'algorithm2_greedy02']
instanceNr = [i for i in range(1, 21)]


def loadAlg(alg):
    if alg in algorithms.keys():
        print("Algo ", alg, " loaded")
        print(algorithms[alg])
        exec(f"from {alg} import {algorithms[alg]} as SOLVEMETHOD")
        solveMethod = SOLVEMETHOD


def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore


def enablePrint():
    sys.stdout = sys.__stdout__


class dummyArgs:
    def __init__(self, instancepath, solutionpath):
        self.instance = instancepath
        self.solution = solutionpath
        self.type = 'txt'
        self.itype = 'txt'


def solveBatch(dir, alg):
    file_path = alg + '.py'
    module_name = alg
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    algModule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(algModule)

    print(dir.strip("./") in os.listdir())
    if dir.strip("./") in os.listdir():
        print("Warning:: Directory already exists. Overwrite?")
        if input('[y/n] :') == 'n':
            return
    else:
        os.mkdir(dir)
    for i in instanceNr:
        print("-" * 50)
        print(i)
        instance = loadInstance(i)
        blockPrint()
        savedPath = algModule.solveAndSave(instance, dir, i)
        enablePrint()
        DoWork(dummyArgs(instancePath(i), savedPath))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch search')
    parser.add_argument('--alg', '-a', metavar='ALGO_NAME',
                        required=True, help='The algorithm name')
    parser.add_argument('--instancenr', '-i', metavar='INSTANCE_FILE',
                        required=True, help='The instance file')
    parser.add_argument('--savedir', '-d',
                        metavar='SAVE_PATH', help='The save location')
    args = parser.parse_args()
    algName = args.alg
    if not algName in algorithms:
        print("Choose one of", algorithms)
        sys.exit()

    # loadAlg(algName)
    #instance = loadInstance(int(args.instancenr))
    instanceNr = [i for i in range(1, int(args.instancenr)+1)]
    print("Solving for up untill ", args.instancenr)
    savedir = args.savedir
    solveBatch(savedir, algName)
