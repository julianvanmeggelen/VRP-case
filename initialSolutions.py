# algorithms that generate initial solution for truck/van routing
from dataClasses import *
from validator.Validate import InstanceCO22

from cachetools import cached

from typing import List, Set, Tuple, Dict

from vrpy import VehicleRoutingProblem
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from util import *
from validator.InstanceCO22 import InstanceCO22
import logging, sys
logging.disable(sys.maxsize) #disable vrpy logging



#####
## Initial van routes using nearest hub heuristic
#####

def distance(loc1: InstanceCO22.Location, loc2: InstanceCO22.Location, ceil: bool = True) -> float:
    dist = math.sqrt((loc1.X - loc2.X)**2 + (loc1.Y - loc2.Y)**2)
    if ceil:
        return math.ceil(dist)
    else:
        return dist


def requestClosestHub(instance: InstanceCO22, request: InstanceCO22.Request) -> int:
    nHubs = len(instance.Hubs)
    hubs = instance.Locations[1:nHubs+1]
    minDist = math.inf
    minDistHubLocID = None
    for i, hub in enumerate(hubs):
        if request.ID in instance.Hubs[i].allowedRequests:
            hubDist = distance(
                instance.Locations[request.customerLocID-1], hub)
            if hubDist < minDist:
                minDist = hubDist
                minDistHubLocID = hub.ID
    return minDistHubLocID


def requestsClosestHub(instance: InstanceCO22) -> dict:
    # return dictionary of {'LOC_ID': ' NEAREST LOC_ID'}
    res = {}
    for req in instance.Requests:
        res[req.ID] = requestClosestHub(instance, req)
    return res


def requestsPerHub(instance: InstanceCO22) -> dict:
    closestHubPerRequest = requestsClosestHub(instance=instance)
    nHubs = len(instance.Hubs)
    hubLocIDs = list(range(2, nHubs+2))
    res = {val: [] for val in hubLocIDs}
    for hubLocID in hubLocIDs:
        for reqID, closestHubLocID in closestHubPerRequest.items():
            if closestHubLocID is hubLocID:
                res[hubLocID].append(reqID)
    return res


def amountPerProduct(instance: InstanceCO22, requests: list) -> list:
    nProducts = len(instance.Products)
    res = [None]*nProducts
    for i in range(nProducts):
        res[i] = sum([req.amounts[i] for req in requests])
    return res


def filterRequests(instance: InstanceCO22, day: int = None, locationsID: int = None) -> list:
    res = instance.Requests.copy()
    if day != None:
        res = [_ for _ in res if _.desiredDay is day]
    if locationsID != None:
        res = [_ for _ in res if _.customerLocID in locationsID]
    return res


def addAllEdges(G: nx.DiGraph, checkWindowOverlap: bool = False) -> nx.DiGraph:
    for locID1, node1 in G.nodes(data=True):
        for locID2, node2 in G.nodes(data=True):
            if locID1 != locID2:
                if checkWindowOverlap and locID1 not in ["Source", "Sink"] and locID2 not in ["Source", "Sink"] and not windowOverlap(node1['periodIDs'], node2['periodIDs']):
                    continue
                dist = math.ceil(
                    math.sqrt(pow(node1['X']-node2['X'], 2) + pow(node1['Y']-node2['Y'], 2)))
                if locID1 == "Sink" or locID2 == "Source" or (locID1 == "Source" and locID2 == "Sink"):
                    continue
                else:
                    G.add_edge(locID1, locID2, time=dist, cost=dist)
    return G


def createNxHub(instance: InstanceCO22, hubLocID: int, requests: list) -> nx.DiGraph:
    G = nx.DiGraph()
    for req in requests:
        reqLoc = instance.Locations[req.customerLocID-1]
        G.add_node(req.ID, locID=reqLoc.ID, reqID=req.ID, X=reqLoc.X,
                   Y=reqLoc.Y, demand=sum(req.amounts), amounts=req.amounts)
    hubLoc = instance.Locations[hubLocID]
    G.add_node("Source", locID=hubLocID, X=hubLoc.X, Y=hubLoc.Y)
    G.add_node("Sink", locID=hubLocID, X=hubLoc.X, Y=hubLoc.Y)
    G = addAllEdges(G)
    return G


def solveHubVRP(instance: InstanceCO22, hubLocID: int, requests: list) -> dict:
    # create networkX
    G = createNxHub(instance, hubLocID, requests)
    G_dict = {i: v for i, v in G.nodes(data=True)}
    # print(G_dict.keys())
    prob = VehicleRoutingProblem(G, load_capacity=instance.VanCapacity)
    prob.duration = instance.VanMaxDistance
    prob.fixed_cost = instance.VanDayCost
    prob.solve()
    best_routes = prob.best_routes
    #best_routes = {id:listReplace(best_routes[id], ["Source","Sink"], hubLocID) for id in best_routes.keys()}
    res = {
        'routes': {key: {'route': [G_dict[id] for id in best_routes[key]]} for key in best_routes.keys()},
        'demand': sum([sum(req.amounts) for req in requests]),
        'amounts': amountPerProduct(instance, requests)
    }
    return res


def parseToHubroutes(res):
    i = 0
    hubRoutesObject = HubRoutes()
    # take dictionary with structure as hubRoutes datamodel
    for day, dayData in res.items():
        dayHubRoutesObject = DayHubRoutes()
        for hubLocID, hubData in dayData.items():
            for routeID, routeData in hubData['routes'].items():
                route = HubRoute(i, hubLocID)
                for node in routeData['route'][1:-1]:  # ignore source and sink
                    nodeObject = RequestNode(
                        reqID=node['reqID'], locID=node['locID'], amounts=node['amounts'], X=node['X'], Y=node['Y'])
                    route.addNode(nodeObject)
                dayHubRoutesObject.addRoute(route)
        hubRoutesObject.addDayHubRoutes(
            day=day, dayHubRoutes=dayHubRoutesObject)
    return hubRoutesObject


@cached(
    cache={},
    key=lambda a: a.CACHE_ID
)

def solveHub(instance: InstanceCO22) -> HubRoutes:
    '''
    Takes instance and returns hubroutes using the nearest hub heuristic
    '''
    nDays = instance.Days
    nHubs = len(instance.Hubs)
    hubLocIDs = list(range(2, nHubs+2))
    hubClusters = requestsPerHub(instance)
    hubRoutes = {}
    for day in range(1, nDays+1):  # hub routing
        dayRoutes = {}
        for hubLocID in hubLocIDs:
            hubCluster = hubClusters[hubLocID]
            #requestsToServe = filterRequests(instance, day, hubCluster)
            requestsToServe = [
                _ for _ in instance.Requests if _.ID in hubCluster and _.desiredDay is day]
            #print(hubCluster, [_.ID for _ in requestsToServe])   #
            if(len(requestsToServe) > 0):
                dayHubRoutes = solveHubVRP(instance, hubLocID, requestsToServe)
                dayRoutes[hubLocID] = dayHubRoutes
        hubRoutes[day] = dayRoutes

    hubRoutes = parseToHubroutes(hubRoutes)
    return hubRoutes

#####
## Initial truck routes using dMin heuristic
#####

def createNxDepot(instance: InstanceCO22, dayRoutes: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    # add hubs
    for (day, hubLocID), hubData in dayRoutes.items():
        G.add_node(f"{hubLocID}.1", locID=hubLocID, demand=0, amounts=0,
                   X=instance.Locations[hubLocID-1].X, Y=instance.Locations[hubLocID-1].Y)

    for (day, hubLocID), hubData in dayRoutes.items():
        i = 1
        nodeID = f"{hubLocID}.{i}"
        while nodeID in G.nodes and G.nodes[nodeID]['demand'] + hubData['demand'] > instance.TruckCapacity:
            i += 1
            nodeID = f"{hubLocID}.{i}"
        if nodeID not in G.nodes:
            G.add_node(nodeID, locID=hubLocID, demand=0, amounts=0,
                       X=instance.Locations[hubLocID-1].X, Y=instance.Locations[hubLocID-1].Y)

        G.nodes[nodeID]['amounts'] = list(
            np.array(G.nodes[nodeID]['amounts']) + np.array(hubData['amounts']))
        G.nodes[nodeID]['demand'] = G.nodes[nodeID]['demand']+hubData['demand']

    G.add_node("Source", locID=1,
               X=instance.Locations[0].X, Y=instance.Locations[0].Y)
    G.add_node("Sink", locID=1,
               X=instance.Locations[0].X, Y=instance.Locations[0].Y)

    G = addAllEdges(G)
    return G

def solveDepotVRP(instance: InstanceCO22, dayRoutes: dict) -> dict:
    # for solving per day
    G = createNxDepot(instance, dayRoutes)
    G_dict = {i: v for i, v in G.nodes(data=True)}
    #print(G_dict.keys())

    prob = VehicleRoutingProblem(G, load_capacity=instance.TruckCapacity)
    prob.duration = instance.TruckMaxDistance
    prob.fixed_cost = instance.TruckDayCost
    prob.solve()
    best_routes = prob.best_routes
    # print(best_routes)
    #best_routes = {id:listReplace(best_routes[id], ["Source","Sink"], hubLocID) for id in best_routes.keys()}
    res = {key: [G_dict[id] for id in best_routes[key]]
           for key in best_routes.keys()}
    # print(res)
    return res

def extractDays(hubRoutes, days: list) -> dict:
    res = {}
    for day, dayRoutes in hubRoutes.items():
        if day in days:
            for hubLocID in dayRoutes:
                newID = (day, hubLocID)
                res[newID] = dayRoutes[hubLocID]
    return res

def solveDepot(instance: InstanceCO22, hubRoutes, useDMin = True) -> dict:
    nDays = instance.Days
    nHubs = len(instance.Hubs)
    hubRoutesDict = hubRoutes.toDict(instance)

    depotRoutes = {}
    
    if useDMin:
        dmin = min([_.daysFresh for _ in instance.Products])
    else: 
        dmin = 1

    for i in range(0, math.ceil(nDays/dmin)):
        periodBegin = dmin*i+1
        periodEnd = dmin*(i+1)+1
        print(i, periodBegin, periodEnd)
        period = list(range(periodBegin, periodEnd))
        periodRoutes = extractDays(hubRoutesDict, period)
        if len(periodRoutes) > 0:  # depot routing
            res = solveDepotVRP(instance, periodRoutes)
            depotRoutes[periodBegin] = res
        else:
            depotRoutes[periodBegin] = {}

    return {'hubRoutes': hubRoutesDict, 'depotRoutes': depotRoutes}


def parseRoutesToDayDepotRoutes(res: dict) -> DayDepotRoutes:
    dayDepotRoutes = DayDepotRoutes()
    for routeID, routeData in res.items():
        hubRoute = DepotRoute(routeID)
        for node in routeData[1:-1]:
            hubNode = HubNode(locID = node['locID'], amounts = node['amounts'], X=node['X'], Y=node['Y'])
            hubRoute.addNode(hubNode)
        dayDepotRoutes.addRoute(hubRoute)
    return dayDepotRoutes

def solveDepotDC(instance: InstanceCO22, hubRoutes, useDMin = True) -> DepotRoutes:
    #returns solution in DepotRoutes format
    nDays = instance.Days
    nHubs = len(instance.Hubs)
    hubRoutesDict = hubRoutes.toDict(instance)

    depotRoutes = DepotRoutes()
    
    if useDMin:
        dmin = min([_.daysFresh for _ in instance.Products])
    else: 
        dmin = 1

    for i in range(0, math.ceil(nDays/dmin)):
        periodBegin = dmin*i+1
        periodEnd = dmin*(i+1)+1
        #print(i, periodBegin, periodEnd)
        period = list(range(periodBegin, periodEnd))
        periodRoutes = extractDays(hubRoutesDict, period)
        if len(periodRoutes) > 0:  # depot routing
            res = solveDepotVRP(instance, periodRoutes)
            dayDepotRoutes = parseRoutesToDayDepotRoutes(res)
            depotRoutes.addDayDepotRoutes(periodBegin, dayDepotRoutes)

    return depotRoutes

def solveDepotDCCached(instance: InstanceCO22, hubRoutes, useDMin = True, cache: DepotRoutes = None, redo: List = None) -> DepotRoutes:
    #returns solution in DepotRoutes format
    nDays = instance.Days
    hubRoutesDict = hubRoutes.toDict(instance)
    depotRoutes = cache
    
    if useDMin:
        dmin = min([_.daysFresh for _ in instance.Products])
    else: 
        dmin = 1

    for i in range(0, math.ceil(nDays/dmin)):
        periodBegin = dmin*i+1
        periodEnd = dmin*(i+1)+1
        if any([(d < periodEnd and d >= periodBegin) for d in redo]): #check if period contains day that needs to be recomputed
            #print(periodBegin)
        #print(i, periodBegin, periodEnd)
            period = list(range(periodBegin, periodEnd))
            periodRoutes = extractDays(hubRoutesDict, period)
            if len(periodRoutes) > 0:  # depot routing
                res = solveDepotVRP(instance, periodRoutes)
                dayDepotRoutes = parseRoutesToDayDepotRoutes(res)
                depotRoutes.depotRoutes[periodBegin] = dayDepotRoutes

    return depotRoutes

def solutionToStr(instance: InstanceCO22, res: dict):
    resultString = "DATASET = CO2022_11 \n \n"

    for day in range(1, instance.Days+1):
        resultString += f"DAY = {day} \n"

        truckString = ""
        if day in res['depotRoutes'].keys():
            nTrucks = len(res['depotRoutes'][day])
            for routeID, truckRoute in res['depotRoutes'][day].items():
                truckString += f"{routeID} "
                for i, hubData in enumerate(truckRoute[1:-1]):
                    amountPerProduct = hubData['amounts']
                    truckString += f"H{hubData['locID'] - 1} {','.join([str(_) for _ in amountPerProduct])} "
                truckString += "\n"
        else:
            nTrucks = 0

        resultString += f"NUMBER_OF_TRUCKS = {nTrucks} \n"
        resultString += truckString

        nVans = 0
        i = 0
        vanString = ""
        for hubLocID in res['hubRoutes'][day].keys():
            for _, route in res['hubRoutes'][day][hubLocID]['routes'].items():
                i += 1
                reqIds = [_['reqID'] for _ in route['route'][1:-1]]
                vanString += f"{i} H{hubLocID-1} {' '.join([str(_) for _ in reqIds])} \n"
            nVans += len(res['hubRoutes'][day][hubLocID]['routes'])
        resultString += f"NUMBER_OF_VANS = {nVans} \n"
        resultString += vanString + "\n"
    return resultString
