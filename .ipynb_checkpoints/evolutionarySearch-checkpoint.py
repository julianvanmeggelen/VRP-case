import initialSolutions
from dataClasses import *
from cachetools import cached
import copy
import time
import random
from typing import List, Set, Tuple, Dict
import pandas as pd
#import llist
import logging
import pprint
from vrpy import VehicleRoutingProblem
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from util import *
from validator.InstanceCO22 import InstanceCO22
import warnings
from collections import OrderedDict
from searchAlgorithms import randomLocalSearch, TreeSearch, EvolutionarySearch, EvolutionarySearchBothEchelons
warnings.filterwarnings("ignore", module="matplotlib\..*")
warnings.filterwarnings("ignore", module="vrpy\..*")
sns.set()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.CRITICAL)

def earlyStopper(costs):
    c = 20
    if len(costs) > c:
        return costs[-1] == costs[-c] 
    return False

def solve(instance, useDMin):
    hubRoutes = initialSolutions.solveHub(instance)
    depotRoutes = initialSolutions.solveDepotDC(instance, hubRoutes)
    initialState = Solution(hubRoutes=hubRoutes, depotRoutes=depotRoutes)
    searcher = EvolutionarySearchBothEchelons(instance = instance, initialState = initialState, generationSize=20, candidateSize=100)
    bestState, bestStateCost = searcher.run(parallel=False, earlyStopping= earlyStopper, recomputeDepotRoute=False, nGenerations=40, useDMin = useDMin) #warmup
    bestState, bestStateCost = searcher.run(parallel=False, earlyStopping= earlyStopper, recomputeDepotRoute=True, nGenerations=100, useDMin = useDMin)
    return bestState


def solveAndSave(instance, path, i):
    #try:
    useDMin = (i not in [6,16])
    res = solve(instance, useDMin)
    #print(res)
    resStr = res.toStr(instance)

    with open("./solutions" + path + f"/solution{i}.txt" ,'w') as file:
        file.write(resStr)

    print("succes")
    return path + f"/solution{i}.txt"
   