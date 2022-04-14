from curses import KEY_LEFT
import math
from validator.InstanceCO22 import InstanceCO22
import matplotlib.pyplot as plt

def instancePath(instanceNr):
    base = "./Instances/"
    folders = ["Instance_1-10/","Instance_11-20/","Instance_21-30/"]
    folder = folders[math.floor((instanceNr-1)/10)]
    path = base + folder + f"Instance_{instanceNr}.txt"
    return path

def loadInstance(instanceNr: int) -> InstanceCO22:
    base = "./Instances/"
    folders = ["Instance_1-10/","Instance_11-20/","Instance_21-30/"]
    folder = folders[math.floor((instanceNr-1)/10)]
    path = base + folder + f"Instance_{instanceNr}.txt"
    return InstanceCO22(inputfile = path, filetype = 'txt')

def plotInstance(instance):
    plt.figure(figsize=(7,7))
    locX = [_.X for _ in instance.Locations]
    locY = [_.Y for _ in instance.Locations]
    nHubs = len(instance.Hubs)
    plt.scatter(locX[0], locY[0], marker=",", label="Depot")
    plt.scatter(locX[1:1+nHubs], locY[1:1+nHubs],marker="^", label="Hub")
    plt.scatter(locX[1+nHubs:], locY[1+nHubs:],marker='.')
    plt.legend()

def plotLocations(locations, ax=None):
    locX = [_.X for _ in locations]
    locY = [_.Y for _ in locations]
    if ax:
        ax.scatter(locX, locY,marker='.')
        ax.legend()
    else:
        plt.figure(figsize=(7,7))
        plt.scatter(locX, locY,marker='.')
        plt.legend()


def listReplace(list, keys, value):
    for i, _ in enumerate(list):
        if _ in keys:
            list[i] = value
    return list


def amountPerProduct(instance, requests):
    nProducts = len(instance.Products)
    res = [None]*nProducts
    for i in range(nProducts):
        res[i] = sum([req.amounts[i] for req in requests])
    return res
    
