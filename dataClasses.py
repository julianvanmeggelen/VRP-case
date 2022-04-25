
from util import *
import numpy as np
import pandas as pd
from typing import List, Set, Tuple, Dict
import random
from validator.InstanceCO22 import InstanceCO22


class DistanceMatrix(object):
    #allows to get distance between to Location IDs or two Request IDs
    
    def __init__(self, instance: InstanceCO22):
        self.locIDs = [_.ID for _ in instance.Locations]
        distances = self._computeDistanceMatrix(instance)
        self.dmDf = pd.DataFrame(index = self.locIDs, columns=self.locIDs, data = distances)

    def _computeDistanceMatrix(self, instance: InstanceCO22, roundUp: bool = True) -> np.ndarray:
        # https://stackoverflow.com/questions/22720864/efficiently-calculating-a-euclidean-distance-matrix-using-numpy
        
        z = np.array([complex(l.X, l.Y) for l in instance.Locations])
        m, n = np.meshgrid(z, z)
        out = abs(m-n)
        if roundUp:
            out = np.ceil(out)
        return out

    def byReqID(self, reqID1, reqID2):
        locID1 = self._getLocID(reqID1)
        locID2 = self._getLocID(reqID2)
        return self.byLocID(locID1, locID2)

    def byLocID(self, locID1, locID2):
        return self.dmDf.loc[locID1, locID2]

    def byReqIDLocID(self, reqID1, locID2):
        locID1 = self._getLocID(reqID1)
        return self.byLocID(locID1, locID2)

    def _getLocID(self, reqID):
        return self.instance.Requests[reqID-1].customerLocID

class Node(object):

    #node carrying request data

    def __init__(self, reqID=None, locID=None, amounts=None, X=None, Y=None, isHub=False):
        # suggest id: requestDI
        self.reqID = reqID
        self.locID = locID
        self.amounts = amounts
        if amounts is not None:
            self.demand = sum(amounts)
        self.isHub = isHub
        self.X = X
        self.Y = Y

    def fromRequest(self, req: InstanceCO22.Request):
        self.reqID = req.ID
        self.locID = req.customerLocID
        self.amounts = req.amounts
        if type(req.amounts) is List:
            self.demand = sum(req.amounts)
        else:
            self.demand = req.amounts
        self.isHub = False
        self.X = None
        self.Y = None
        return self

    def __repr__(self):
        return f"Node(reqID:{self.reqID}, locID:{self.locID})"

class HubRoute(object):

    def __init__(self, routeID, hubLocID):
        self.nodes = []
        self.routeID = routeID
        self.hubLocID = hubLocID
        self.cachedDistance = None

    def addNode(self, node: Node):
        # add node to end of route
        self.nodes.append(node)
        self.cachedDistance = None
        return self

    def addNodes(self, nodes: List[Node]):
        for node in nodes:
            self.addNode(node)
        return self

    def insertNodeAfter(self, i, newNode: Node):
        # insert node after specified index
        #appendAfterIndex = self._findIndexOfID(reqID)
        self.nodes.insert(i+1, newNode)
        self.cachedDistance = None
        return self

    def insertNodeBefore(self, i, newNode: Node):
        # insert node after specified index
        #appendBeforeIndex = self._findIndexOfID(reqID)
        self.nodes.insert(i, newNode)
        self.cachedDistance = None
        return self

    def deleteNode(self, node):
        self.nodes.remove(node)

    def deleteNodes(self, nodes: List[Node]):
        for node in nodes:
            self.deleteNode(node)

    def deleteNodeByReqID(self, reqID):
        indexToDelete = self._findIndexOfID(reqID)
        self.nodes = [j for i, j in enumerate(self.nodes) if i != indexToDelete ]

    def replaceNode(self, reqID, newNode: Node):
        indexToReplace = self._findIndexOfID(reqID)
        self.nodes[indexToReplace] = newNode
        self.cachedDistance = None
        return self

    def swapNodes(self, ind1, ind2):
        self.nodes[ind1], self.nodes[ind2] = self.nodes[ind2], self.nodes[ind1]
        self.cachedDistance = None
        return self

    def _findIndexOfID(self, nodeID):
        i = 0
        el = self.nodes[i]

        while el.reqID != nodeID:
            i += 1
            el = self.nodes[i]

        return i

    def length(self, distanceMatrix: DistanceMatrix): #distance length
        if self.cachedDistance:
            return self.cachedDistance
        
        #distance between nodes (clients)
        dist = 0
        for i in range(0, len(self.nodes)-1):
            dist += distanceMatrix.byLocID(self.nodes[i].locID, self.nodes[i+1].locID)

        #distance from hub to first node and from last node to hub
        dist += distanceMatrix.byLocID(self.hubLocID, self.nodes[0].locID)
        dist += distanceMatrix.byLocID(self.hubLocID, self.nodes[-1].locID)
        self.cachedDistance = dist
        return dist

    def demand(self):
        return sum(_.demand for _ in self.nodes)

    def amounts(self):
        nProducts = len(self.nodes[0].amounts)
        res = []
        for i in range(nProducts):
            res.append(sum([_.amounts[i] for _ in self.nodes]))
        return res

    def nProducts(self):
        return len(self.nodes[0].amounts)


    def __repr__(self):
        return f"HubRoute from hub {self.hubLocID}: " + " -> ".join([_.__repr__() for  _ in self.nodes])

    def __len__(self): #nodes (except hub)
        return len(self.nodes)
    
    def __getitem__(self, i):
        return self.nodes[i]

class DayHubRoutes(object):
    def __init__(self):
        self.routes = []

    def addRoute(self, route: HubRoute):
        self.routes.append(route)

    def addRouteFromNodes(self, routeID, hubLocID, nodes: List[Node]):
        route = HubRoute(routeID, hubLocID).addNodes(nodes)
        self.routes.append(route)
        self.hubsUsed.add(hubLocID)
        # nodes: [nodeID1, nodeID2, nodeID3, nodeID4]

    def computeCost(self, vanDistanceCost, vanDayCost, distanceMatrix: DistanceMatrix):
        # compute cost of distance and #vans
        cost = 0
        for route in self.routes:
            cost += route.length(distanceMatrix) * vanDistanceCost
            cost += vanDayCost
        return cost

    def coversRequestIDs(self, requestIDs: List[int], verbose=False) -> bool:
        reqsToServe = requestIDs.copy()
        for route in self.routes:
            for node in route.nodes:
                reqsToServe.remove(node.reqID) if node.reqID in reqsToServe else None
        allServed = (len(reqsToServe) == 0)
        if not allServed and verbose:
            print(allServed, "not served")
        return allServed

    def getRoutesFromHub(self, hubLocID) -> List[HubRoute]:
        res = []
        for route in self.routes:
            if route.hubLocID is hubLocID:
                res.append(route)
        return res

    def nHubs(self):
        return len(self.hubsUsed())
    
    def hubsUsed(self):
        res = set()
        for route in self.routes:
            res.add(route.hubLocID)
        return res

    def isFeasible(self, vanCapacity, vanMaxDistance, distanceMatrix, verbose=False) -> bool:
        # check max capacity and maxdistance constraints. None of the operators will remove nodes from the routes so it is assumed that
        # if all requests are served in the initial solution this will be the case for all derived solutions.
        notOverCapacity = True
        notOverDistance = True
        for route in self.routes:
            if route.demand() > vanCapacity:
                if verbose:
                    print(
                        f"Route with id {route.routeID} has a demand of {route.demand()} > vanCapacity = {vanCapacity}")
                notOverCapacity = False
            routeLength = route.length(distanceMatrix)
            if routeLength > vanMaxDistance:
                if verbose:
                    print(
                        f"Route with id {route.routeID} has a length of {routeLength} > vanMaxDistance = {vanMaxDistance}")
                notOverDistance = False
        feasible = notOverCapacity and notOverDistance
        return feasible

    def __repr__(self):
        res = ""
        for route in self.routes:
            res+= route.__repr__() + "\n"
        return res
    
    def __len__(self):
        return len(self.routes)

    # operators for neighbourhood exploration
    #   Based on Kuo, Wang

    def randomNodeInsertion(self):
        # Delete node from a route and insert inbetween two succesive nodes (in any route)

        r1 = random.choice(self.routes)     #choose random route r1
        n1 = random.choice(r1)              #choose random node n1 from route r1
        r2 = random.choice(self.routes)     #choose next random route r2
        #print(n2.reqID)   
        #print(r2.nodes[0].reqID)
        r1.deleteNode(n1)  
        insertBefore = random.randint(0,len(r2)+1)            #choose next random node n2             
        r2.insertNodeBefore(insertBefore, n1)    #insert n1 in route r2 after node n2
        if len(r1) == 0:
            self.routes.remove(r1)
        return self

    def randomSectionInsertion(self):
        r1 = random.choice(self.routes)     #choose random route r1
        n = len(r1)
        sectionBegin = random.randint(0,n-1)
        sectionEnd = random.randint(sectionBegin,n)
        section = r1.nodes[sectionBegin:sectionEnd]
        r1.deleteNodes(section)
        r2 = random.choice(self.routes)     #choose next random route r2
        insertBefore = random.randint(0,len(r2)+1)            #choose next random node n2  
        section.reverse()  
        for node in section:
            r2.insertNodeBefore(insertBefore, node)
        if len(r1) == 0:
            self.routes.remove(r1)
        return self


    def shuffleRoute(self):
        #shuffle two nodes within one route
        r1 = random.choice(self.routes)     #choose random route r1
        if(len(r1) == 1):
            return self
        node1 = random.randint(0,len(r1)-1)            #choose next random node n2  
        node2 = random.randint(0,len(r1)-1)   
        while node2 == node1:
            node2 = random.randint(0,len(r1)-1)  
        #print(r1)
        r1.swapNodes(node1, node2)    
        #print(r1)
        return self

    def randomMergeRoutes(self):
        # take two routes and paste 1 route to the back of the other
        if len(self) > 1:
            r1 = random.choice(self.routes) 
            self.routes.remove(r1)
            r2 = random.choice(self.routes)   
            r2.addNodes(r1.nodes)
        return self

    def randomNodeExchange():
        # Select two nodes and switch position (can be in different routes)
        raise NotImplementedError

    def randomArcExchange():
        # Select two pairs of nodes and exchange positions
        raise NotImplementedError

    def randomMoveRouteToHub():
        raise NotImplementedError
    
    def randomMoveRoutesToNearestHub():
        raise NotImplementedError

    def randomSectionExchange():
        # Select two sections from two different routes and exchange sections

        raise NotImplementedError

class HubRoutes(object):
    def __init__(self):
        self.hubRoutes = {}
        self.deliverEarlyPenalty = 0
        
    def addDayHubRoutes(self,day, dayHubRoutes: DayHubRoutes):
        self.hubRoutes[day] = dayHubRoutes

    def computeCost(self, vanDistanceCost, vanDayCost, vanCost, distanceMatrix: DistanceMatrix):
        cost = 0
        cost += self.deliverEarlyPenalty

        for day, dayHubRoutes in self.hubRoutes.items():
            cost += dayHubRoutes.computeCost(vanDistanceCost, vanDayCost, distanceMatrix)
        cost += self.nVansRequired() * vanCost

        return cost

    def nVansRequired(self):
        return max([len(dayHubRoutes) for day, dayHubRoutes in self.hubRoutes.items()])
    
    def toDict(self, instance):
        res = {}
        for day, dayHubRoutes in self.hubRoutes.items():
            dayres = {}
            for locID in dayHubRoutes.hubsUsed():
                dayres[locID] = {}
                hub = instance.Locations[locID]
                hubNode = {'X': hub.X,'Y':hub.Y,'locID': locID}
                hubRoutes = dayHubRoutes.getRoutesFromHub(locID)
                nProducts = hubRoutes[0].nProducts()
                dayres[locID]['routes'] = {}
                i = 0
                for route in hubRoutes:
                    dayres[locID]['routes'][i] = {}
                    dayres[locID]['routes'][i]['route'] = [hubNode] + [{'X': n.X,'Y':n.Y,'demand':n.demand,'locID':n.locID,'reqID':n.reqID} for n in route.nodes] + [hubNode]
                    i+=1
                dayres[locID]['amounts'] = [sum([route.amounts()[i] for route in hubRoutes]) for i in range(nProducts)]
                dayres[locID]['demand'] = sum([route.demand() for route in hubRoutes])
            res[day] = dayres
        return res
        
    def __repr__(self):
        res = ""
        for day, data in self.hubRoutes.items():
            res += "-"*10 + f"day {day}" + "-"*10 + "\n"
            res += data.__repr__() + "\n"
        return res

    def __getitem__(self,i):
        return self.hubRoutes[i]

    # operators for neighbourhood exploration

