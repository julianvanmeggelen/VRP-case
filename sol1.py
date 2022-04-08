from validator.InstanceCO22 import InstanceCO22
from util import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from vrpy import VehicleRoutingProblem
import argparse
import sys
import os



# [1] Assign clients to hubs
# [2] For each day:
#   [1] Solve per hub
#   [2] Determine hub product requirements
#   [3] Solve for depot


def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def computeDistanceMatrix(instance: InstanceCO22, roundUp: bool = True) -> np.ndarray:
    # https://stackoverflow.com/questions/22720864/efficiently-calculating-a-euclidean-distance-matrix-using-numpy
        
    z = np.array([complex(l.X, l.Y) for l in instance.Locations])
    m, n = np.meshgrid(z, z)
    out = abs(m-n)
    if roundUp:
        out = np.ceil(out)
    return out

def assignHub(distanceMatrix: np.ndarray, hubs: list, nDepot: int = 1)-> np.ndarray:
    # Take distance matrix and assign to each point the closest hub 
    # hubs: list of indices containing place of the hubs in the distanceMatrix
    # Returns array with hub indice for each location (starting with 0). Includes depot and hubs

    hubDistanceMatrix =  distanceMatrix[hubs]
    assignedHub =  np.argmin(hubDistanceMatrix, axis=0)
    return assignedHub

def pointsPerHub(assignedHub: np.ndarray) -> np.ndarray:
        #List of indices per hub
    sort_idx = np.argsort(assignedHub)
    a_sorted = assignedHub[sort_idx]
    unq_first = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    unq_count = np.diff(np.nonzero(unq_first)[0])
    unq_idx = np.split(sort_idx, np.cumsum(unq_count))
    return unq_idx

def filterRequests(instance, day = None, locationsID = None):
    res = instance.Requests.copy()
    if day != None:
        res = [_ for _ in res if _.desiredDay is day]
    if locationsID != None:
        res = [_ for _ in res if _.customerLocID in locationsID]
    return res

def locationsWithRequest(locations, requests):
    loc_ids = [_.customerLocID for _ in requests]
    return [_ for _ in locations if _.ID in loc_ids]

def amountPerProduct(instance, requests):
    nProducts = len(instance.Products)
    res = [None]*nProducts
    for i in range(nProducts):
        res[i] = sum([req.amounts[i] for req in requests])
    return res

def requestsIdForRoute(route, instance, day):
    #Reverse engineer request ids for given route and day
    route = route[1:-1] #trim hub
    res = []
    requests = filterRequests(instance, day=day)
    for i, locID in enumerate(route):
        for req in requests:
            if req.customerLocID == locID:
                res.append(req.ID)
    return res

def toNetworkX_hubschedule(clientLocations, requests, hub=None):
    locations = clientLocations
    locations = locations + [hub]
    G = nx.DiGraph()
    for loc in locations:
        if loc.ID is hub.ID:
            G.add_node("Source")
            G.add_node("Sink")
        else:
            G.add_node(loc.ID)
            
    for req in requests:
        if req in G.nodes.keys():
            G.nodes[req.customerLocID]['demand'] = 0

    for req in requests:
        print(req.ID, " served")
        if req in G.nodes.keys():
            G.nodes[req.customerLocID]['demand'] =  G.nodes[req.customerLocID]['demand'] + sum(req.amounts)

    for l1 in locations:
        for l2 in locations:
            if l1.ID != l2.ID:
                dist = math.ceil( math.sqrt( pow(l1.X-l2.X,2) + pow(l1.Y-l2.Y,2) ))
                if l1.ID != hub.ID and l2.ID != hub.ID: 
                    G.add_edge(l1.ID, l2.ID, time = dist, cost=dist)
                elif l1.ID == hub.ID:
                    G.add_edge("Source", l2.ID, time = dist, cost=dist)
                elif l2.ID == hub.ID:
                    G.add_edge(l1.ID, "Sink", time = dist, cost=dist)
    pos = {_.ID:[_.X,_.Y] for _ in locations}
    pos['Source'] = [hub.X,hub.Y]
    pos['Sink'] = [hub.X,hub.Y]
    return G, pos

def toNetworkX_depotschedule(hubs, depot, demands):
    locations = hubs
    locations = locations + [depot]
    G = nx.DiGraph()
    for loc in locations:
        if loc.ID is depot.ID:
            G.add_node("Source")
            G.add_node("Sink")
        else:
            G.add_node(loc.ID)

    for ID, v in demands.items():
        G.nodes[ID]['demand'] = v['demand']

    for l1 in locations:
        for l2 in locations:
            if l1.ID != l2.ID:
                dist = math.ceil( math.sqrt( pow(l1.X-l2.X,2) + pow(l1.Y-l2.Y,2) ))
                if l1.ID != depot.ID and l2.ID != depot.ID: 
                    G.add_edge(l1.ID, l2.ID, time = dist, cost=dist)
                elif l1.ID == depot.ID:
                    G.add_edge("Source", l2.ID, time = dist, cost=dist)
                elif l2.ID == depot.ID:
                    G.add_edge(l1.ID, "Sink", time = dist, cost=dist)
    pos = {_.ID:[_.X,_.Y] for _ in locations}
    pos['Source'] = [depot.X,depot.Y]
    pos['Sink'] = [depot.X,depot.Y]
    return G, pos

def sol1(instance):
    #hub routes
    distanceMatrix = computeDistanceMatrix(instance,roundUp=True)
    nHubs = len(instance.Hubs)
    hubs = [_ for _ in range(1,nHubs+1)]
    assignedHub = assignHub(distanceMatrix, hubs=hubs)
    hubs = pointsPerHub(assignedHub)

    #Hub schedule
    hubRoutes = {}
    for day in range(1,instance.Days+1):
        print("---------------------")
        dayRoutes = {}
        for i, hub_locations_ID in enumerate(hubs):
            hub_ID = i + 2
            requests = filterRequests(instance, day=day, locationsID = list(hub_locations_ID+1))
            if len(requests) == 0:
                continue
            locations = locationsWithRequest(instance.Locations, requests)
            print(day)
            print(len(requests))
            print(len(locations))
            G, _ = toNetworkX_hubschedule(clientLocations = locations, requests=requests, hub=instance.Locations[i+1]) #add 1 because 1 depot
            prob = VehicleRoutingProblem(G, load_capacity=instance.VanCapacity)
            prob.duration = instance.VanMaxDistance
            prob.solve()
            best_routes = prob.best_routes
            best_routes = {'routes':{id:listReplace(best_routes[id], ["Source","Sink"], hub_ID) for id in best_routes.keys()}}
            best_routes['demand'] = sum([sum(req.amounts) for req in requests])
            best_routes['demandPerProduct'] = amountPerProduct(instance, requests)
            dayRoutes[hub_ID] = best_routes
        hubRoutes[day] = dayRoutes

    #depot schedule
    depotRoutes = {}
    depotLocation = instance.Locations[0]
    depotID = 1
    for day in range(1,instance.Days+1):
        hubsUsed = hubRoutes[day]
        hubLocations = [instance.Locations[_-1] for _ in hubsUsed.keys()]
        if len(hubLocations) > 0:
            G, _ = toNetworkX_depotschedule(hubLocations, depotLocation, hubsUsed)
            prob = VehicleRoutingProblem(G, load_capacity=instance.TruckCapacity)
            prob.duration = instance.TruckMaxDistance
            prob.solve()
            depotRoutes[day] = {id:listReplace(prob.best_routes[id], ["Source","Sink"], depotID) for id in prob.best_routes.keys()}
        else:
            depotRoutes[day] = {}
   
    return {'hubRoutes':hubRoutes, 'depotRoutes':depotRoutes}


def toStr(res, instance):
    resultString = "DATASET = CO2022_11 \n \n"
    for day in range(1, instance.Days+1):
        resultString += f"DAY = {day} \n"
        nTrucks = len(res['depotRoutes'][day])
        resultString += f"NUMBER_OF_TRUCKS = {nTrucks} \n"

        truckString = ""
        for i, truckRoute in res['depotRoutes'][day].items():
            truckString += f"{i} "
            for hub in truckRoute[1:-1]:
                amountPerProduct = res['hubRoutes'][day][hub]['demandPerProduct']
                truckString += f"H{hub-1} {','.join([str(_) for _ in amountPerProduct])} "
            truckString += "\n"
        resultString += truckString

        nVans = 0
        i = 0
        vanString = ""
        for hub in res['hubRoutes'][day].keys():
            for _, route in res['hubRoutes'][day][hub]['routes'].items():
                i+=1
                reqIds = requestsIdForRoute(route, instance, day)
                vanString += f"{i} H{hub-1} {' '.join([str(_) for _ in reqIds])} \n"
            nVans += len(res['hubRoutes'][day][hub]['routes'])
        resultString += f"NUMBER_OF_VANS = {nVans} \n"  
        resultString += vanString + "\n"
    return resultString
    
def solveAndSave(instance, path, i):
    blockPrint()
    try:
        res = sol1(instance)
        resStr = toStr(res, instance)
        with open(path + f"/solution{i}.txt" ,'w') as file:
            file.write(resStr)
        print("succes")
        return path + f"/solution{i}.txt"
    except Exception as e:
        print(e)

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Naive algo')
        parser.add_argument('--instancenr', '-i', metavar='INSTANCE_FILE', required=True, help='The instance file')
        parser.add_argument('--savedir', '-d', metavar='SAVE_PATH', help='The save location')
        args = parser.parse_args()
        instance = loadInstance(int(args.instancenr))
        solveAndSave(instance, args.savedir, args.instancenr)










