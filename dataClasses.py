
from re import L
from util import *
import numpy as np
import pandas as pd
from typing import List, Set, Tuple, Dict
import random
from validator.InstanceCO22 import InstanceCO22
import copy
import initialSolutions


class DistanceMatrix(object):
    # allows to get distance between to Location IDs or two Request IDs

    def __init__(self, instance: InstanceCO22):
        self.locIDs = [_.ID for _ in instance.Locations]
        distances = self._computeDistanceMatrix(instance)
        self.instance = instance
        self.dmDf = pd.DataFrame(
            index=self.locIDs, columns=self.locIDs, data=distances)

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
        # use this to find distance between request locIDs
        return self.dmDf.loc[locID1, locID2]

    def byReqIDLocID(self, reqID1, locID2):
        # use this to find distance between one requestID and one LocID
        # for distance between hub and request use reqToHubDist
        locID1 = self._getLocID(reqID1)
        return self.byLocID(locID1, locID2)

    def reqToHubDist(self, reqID, locID, hubLocID):
        # Also checks if hub can serve requestID, and return math.inf if not
        hub = self.instance.Hubs[hubLocID-2]
        if reqID not in hub.allowedRequests:
            return math.inf
        else:
            return self.byLocID(locID, hubLocID)

    def _getLocID(self, reqID):
        return self.instance.Requests[reqID-1].customerLocID


class RequestNode(object):

    # node carrying request data

    def __init__(self, reqID=None, locID=None, amounts=None, X=None, Y=None, isHub=False):
        # suggest id: requestDI
        self.reqID = reqID
        self.locID = locID
        self.amounts = amounts
        self.daysDeliveredEarly = 0
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
        self.cachedHubCanServe = None

    def addNode(self, node: RequestNode):
        # add node to end of route
        self.nodes.append(node)
        self.cachedDistance = None
        self.cachedHubCanServe = None
        return self

    def addNodes(self, nodes: List[RequestNode]):
        for node in nodes:
            self.addNode(node)
        return self

    def insertNodeAfter(self, i, newNode: RequestNode):
        # insert node after specified index
        #appendAfterIndex = self._findIndexOfID(reqID)
        self.nodes.insert(i+1, newNode)
        self.cachedDistance = None
        self.cachedHubCanServe = None

        return self

    def insertNodeBefore(self, i, newNode: RequestNode):
        # insert node after specified index
        #appendBeforeIndex = self._findIndexOfID(reqID)
        self.nodes.insert(i, newNode)
        self.cachedDistance = None
        self.cachedHubCanServe = None

        return self

    def deleteNode(self, node):
        self.nodes.remove(node)
        self.cachedDistance = None
        self.cachedHubCanServe = None

    def deleteNodes(self, nodes: List[RequestNode]):
        for node in nodes:
            self.deleteNode(node)
        self.cachedDistance = None
        self.cachedHubCanServe = None

    def deleteNodeByReqID(self, reqID):
        indexToDelete = self._findIndexOfID(reqID)
        self.nodes = [j for i, j in enumerate(
            self.nodes) if i != indexToDelete]
        self.cachedDistance = None
        self.cachedHubCanServe = None

    def replaceNode(self, reqID, newNode: RequestNode):
        indexToReplace = self._findIndexOfID(reqID)
        self.nodes[indexToReplace] = newNode
        self.cachedDistance = None
        self.cachedHubCanServe = None
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

    def checkHubCanServe(self, instance):
        # if self.cachedHubCanServe:
        #    return self.cachedHubCanServe

        reqIDs = set([_.reqID for _ in self.nodes])
        hubAllowedReqIDs = set(instance.Hubs[self.hubLocID-2].allowedRequests)
        hubsCanServe = len(reqIDs-hubAllowedReqIDs) == 0
        #self.cachedHubCanServe = hubsCanServe
        return hubsCanServe

    def length(self, distanceMatrix: DistanceMatrix):  # distance length
        #if self.cachedDistance:
            #return self.cachedDistance

        # distance between nodes (clients)
        dist = 0
        for i in range(0, len(self.nodes)-1):
            dist += distanceMatrix.byLocID(
                self.nodes[i].locID, self.nodes[i+1].locID)

        # distance from hub to first node and from last node to hub
        dist += distanceMatrix.reqToHubDist(
            self.nodes[0].reqID, self.nodes[0].locID, self.hubLocID)
        dist += distanceMatrix.reqToHubDist(
            self.nodes[-1].reqID, self.nodes[-1].locID, self.hubLocID)
        #self.cachedDistance = dist
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

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return f"HubRoute from hub {self.hubLocID}: " + " -> ".join([_.__repr__() for _ in self.nodes])

    def __len__(self):  # nodes (except hub)
        return len(self.nodes)

    def __getitem__(self, i):
        return self.nodes[i]


class DayHubRoutes(object):
    def __init__(self):
        self.routes = []

    def addRoute(self, route: HubRoute):
        self.routes.append(route)

    def addRouteFromNodes(self, routeID, hubLocID, nodes: List[RequestNode]):
        route = HubRoute(routeID, hubLocID).addNodes(nodes)
        self.routes.append(route)
        self.hubsUsed.add(hubLocID)
        # nodes: [nodeID1, nodeID2, nodeID3, nodeID4]

    def computeCost(self, vanDistanceCost, vanDayCost, distanceMatrix: DistanceMatrix, useHubOpeningCost=False, instance=None):
        # compute cost of distance and #vans
        cost = 0
        for route in self.routes:
            cost += route.length(distanceMatrix) * vanDistanceCost
            cost += vanDayCost

        if useHubOpeningCost:
            hubs = self.hubsUsed()
            for hub in hubs:
                cost += instance.Hubs[hub-2].hubOpeningCost
        return cost

    def coversRequestIDs(self, requestIDs: List[int], verbose=False) -> bool:
        reqsToServe = requestIDs.copy()
        for route in self.routes:
            for node in route.nodes:
                reqsToServe.remove(
                    node.reqID) if node.reqID in reqsToServe else None
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

    def replaceHub(self, hubLocIDOld, hubLocIDNew):
        for route in self.routes:
            if route.hubLocID == hubLocIDOld:
                route.hubLocID = hubLocIDNew
        return self

    def isFeasible(self, vanCapacity, vanMaxDistance, distanceMatrix, instance, verbose=False, useCheckHubCanServe=False) -> bool:
        # check max capacity and maxd istance constraints. None of the operators will remove nodes from the routes so it is assumed that
        # if all requests are served in the initial solution this will be the case for all derived solutions.
        notOverCapacity = True
        notOverDistance = True
        hubCanServeRequests = True
        for route in self.routes:
            if useCheckHubCanServe:
                if not route.checkHubCanServe(instance):
                    hubCanServeRequests = False
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
        feasible = notOverCapacity and notOverDistance and hubCanServeRequests
        return feasible

    def __repr__(self):
        res = ""
        for route in self.routes:
            res += route.__repr__() + "\n"
        return res

    def __len__(self):
        return len(self.routes)

    # operators for neighbourhood exploration
    #   Based on Kuo, Wang

    def randomNodeInsertion(self):
        # Delete node from a route and insert inbetween two succesive nodes (in any route)

        r1 = random.choice(self.routes)  # choose random route r1
        n1 = random.choice(r1)  # choose random node n1 from route r1
        r2 = random.choice(self.routes)  # choose next random route r2
        # print(n2.reqID)
        # print(r2.nodes[0].reqID)
        r1.deleteNode(n1)
        # choose next random node n2
        insertBefore = random.randint(0, len(r2)+1)
        # insert n1 in route r2 after node n2
        r2.insertNodeBefore(insertBefore, n1)
        if len(r1) == 0:
            self.routes.remove(r1)
        return self

    def randomSectionInsertion(self):
        r1 = random.choice(self.routes)  # choose random route r1
        n = len(r1)
        sectionBegin = random.randint(0, n-1)
        sectionEnd = random.randint(sectionBegin, n)
        section = r1.nodes[sectionBegin:sectionEnd]
        r1.deleteNodes(section)
        r2 = random.choice(self.routes)  # choose next random route r2
        # choose next random node n2
        insertBefore = random.randint(0, len(r2)+1)
        section.reverse()
        for node in section:
            r2.insertNodeBefore(insertBefore, node)
        if len(r1) == 0:
            self.routes.remove(r1)
        return self

    def shuffleRoute(self):
        # shuffle two nodes within one route
        r1 = random.choice(self.routes)  # choose random route r1
        if(len(r1) == 1):
            return self
        node1 = random.randint(0, len(r1)-1)  # choose next random node n2
        node2 = random.randint(0, len(r1)-1)
        while node2 == node1:
            node2 = random.randint(0, len(r1)-1)
        # print(r1)
        r1.swapNodes(node1, node2)
        # print(r1)
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
        self.deliverEarlyPenalties = {}  # {nodeid: penaltu}
        self.costCaching = {}
        self.hubsUsedCaching = {}

    def addDayHubRoutes(self, day, dayHubRoutes: DayHubRoutes):
        self.hubRoutes[day] = dayHubRoutes
        self.costCaching[day] = None
        self.hubsUsedCaching[day] = None

    def isFeasible(self, VanCapacity, VanMaxDistance, dm, instance, verbose=False, useCheckHubCanServe=False):
        for day, dayHubRoutes in self.hubRoutes.items():
            if not dayHubRoutes.isFeasible(VanCapacity, VanMaxDistance, dm, instance, verbose, useCheckHubCanServe):
                return False
        return True

    def computeCost(self, vanDistanceCost, vanDayCost, vanCost, deliverEarlyPenalty, distanceMatrix: DistanceMatrix, useHubOpeningCost=False, instance=None):
        cost = 0

        for day, dayHubRoutes in self.hubRoutes.items():
            if self.costCaching[day]:
                cost += self.costCaching[day]
            else:
                dayCost = dayHubRoutes.computeCost(
                    vanDistanceCost, vanDayCost, distanceMatrix)
                cost += dayCost
                #self.costCaching[day] = dayCost

        cost += self.nVansRequired() * vanCost

        cost += self.deliverEarlyPenalty(cost=deliverEarlyPenalty)

        if useHubOpeningCost:
            hubs = self.hubsUsed()  # hubopeningcosts
            for hub in hubs:
                cost += instance.Hubs[hub-2].hubOpeningCost
        return cost

    def deliverEarlyPenalty(self, cost):
        res = 0
        for reqID, daysEarly in self.deliverEarlyPenalties.items():
            res += cost**(daysEarly)
        return res

    def hubsUsed(self) -> Set:
        res = set()
        for day, dayHubRoutes in self.hubRoutes.items():
            if self.hubsUsedCaching[day]:
                res = res.union(self.hubsUsedCaching[day])
            else:
                dayHubRoutesHubsUsed = dayHubRoutes.hubsUsed()
                self.hubsUsedCaching[day] = dayHubRoutesHubsUsed
                res = res.union(dayHubRoutesHubsUsed)
        return res

    def nVansRequired(self):
        return max([len(dayHubRoutes) for day, dayHubRoutes in self.hubRoutes.items()])

    def days(self):
        return list(self.hubRoutes.keys())

    def toDict(self, instance):
        res = {}
        for day, dayHubRoutes in self.hubRoutes.items():
            dayres = {}
            for locID in dayHubRoutes.hubsUsed():
                dayres[locID] = {}
                hub = instance.Locations[locID]
                hubNode = {'X': hub.X, 'Y': hub.Y, 'locID': locID}
                hubRoutes = dayHubRoutes.getRoutesFromHub(locID)
                nProducts = hubRoutes[0].nProducts()
                dayres[locID]['routes'] = {}
                i = 0
                for route in hubRoutes:
                    dayres[locID]['routes'][i] = {}
                    dayres[locID]['routes'][i]['route'] = [
                        hubNode] + [{'X': n.X, 'Y': n.Y, 'demand': n.demand, 'locID': n.locID, 'reqID': n.reqID} for n in route.nodes] + [hubNode]
                    i += 1
                dayres[locID]['amounts'] = [
                    sum([route.amounts()[i] for route in hubRoutes]) for i in range(nProducts)]
                dayres[locID]['demand'] = sum(
                    [route.demand() for route in hubRoutes])
            res[day] = dayres
        return res

    def __repr__(self):
        res = ""
        for day, data in self.hubRoutes.items():
            res += "-"*10 + f"day {day}" + "-"*10 + "\n"
            res += data.__repr__() + "\n"
        return res

    def __getitem__(self, i):
        return self.hubRoutes[i]

    def copy(self):
        return copy.deepcopy(self)

    # operators for neighbourhood exploration

    def applyOperator(self, day, operatorID):
        # apply intraday operator

        dayHubRoute = self.hubRoutes[day]

        operators = [
            dayHubRoute.randomMergeRoutes,
            dayHubRoute.randomNodeInsertion,
            dayHubRoute.randomSectionInsertion,
            dayHubRoute.shuffleRoute
        ]

        if operatorID > len(operators)-1:
            print("operatorID out of range")
            return

        if len(dayHubRoute) > 0:
            operatorToApply = operators[operatorID]
            operatorToApply()
            self.costCaching[day] = None
            self.hubsUsedCaching[day] = None

    # intraday operators
    def intraDayRandomMergeRoutes(self, day):
        dayHubRoute = self.hubRoutes[day]
        if len(dayHubRoute) > 0:
            dayHubRoute.randomMergeRoutes()
            self.costCaching[day] = None
            self.hubsUsedCaching[day] = None

    def intraDayRandomNodeInsertion(self, day):
        dayHubRoute = self.hubRoutes[day]
        if len(dayHubRoute) > 0:
            dayHubRoute.randomMergeRoutes()
            self.costCaching[day] = None
            self.hubsUsedCaching[day] = None

    def intraDayRandomSectionInsertion(self, day):
        dayHubRoute = self.hubRoutes[day]
        if len(dayHubRoute) > 0:
            dayHubRoute.randomSectionInsertion()
            self.costCaching[day] = None
            self.hubsUsedCaching[day] = None

    def intraDayShuffleRoute(self, day):
        dayHubRoute = self.hubRoutes[day]
        if len(dayHubRoute) > 0:
            dayHubRoute.shuffleRoute()
            self.costCaching[day] = None
            self.hubsUsedCaching[day] = None

    # Multiday operators

    def randomMoveNodeDayEarly(self, day=None):
        # move random node to a day earlier to
        if day == None:
            day = random.randint(2, 20)

            while len(self.hubRoutes[day]) < 1:
                day = random.randint(2, 20)

        r1 = random.choice(self.hubRoutes[day].routes)

        # print(r1)
        n1 = random.choice(r1.nodes)
        # print(n1)

        if len(self.hubRoutes[day-1]) == 0:
            return self

        r2 = random.choice(self.hubRoutes[day-1].routes)
        # print('r2before',r2)
        n1.daysDeliveredEarly += 1
        r1.deleteNode(n1)
        insertBefore = random.randint(0, len(r2)+1)
        r2.insertNodeBefore(insertBefore, n1)
        # print('r2after',r2)

        if len(r1) == 0:
            self.hubRoutes[day].routes.remove(r1)

        self.deliverEarlyPenalties[n1.reqID] = n1.daysDeliveredEarly

        self.costCaching[day] = None
        self.costCaching[day-1] = None
        self.hubsUsedCaching[day] = None
        return self

    def randomChooseOtherHub(self, allHubs: set):
        # transfer all routes of one hub to another hub
        hubsUsed = self.hubsUsed()
        if len(hubsUsed) > 1:
            h1 = random.choice(list(hubsUsed))
            h2 = random.choice(list(allHubs.difference([h1])))

            for day, dayHubRoutes in self.hubRoutes.items():
                dayHubRoutes.replaceHub(h1, h2)
                self.hubsUsedCaching[day] = None
        return self


class HubNode(object):
    def __init__(self, locID=None, amounts=None, X=None, Y=None):
        self.locID = locID
        self.amounts = amounts
        self.daysDeliveredEarly = 0
        if amounts is not None:
            self.demand = sum(amounts)
        self.X = X
        self.Y = Y

    def __repr__(self):
        return f"HubNode(locID:{self.locID})"


class DepotRoute(object):
    def __init__(self, routeID):
        self.nodes: List[HubNode] = []
        self.routeID = routeID
        self.depotLocID = 1  # depot is always locID 1
        self.cachedDistance = None
        self.cachedHubCanServe = None

    def addNode(self, node: HubNode):
        # add node to end of route
        self.nodes.append(node)
        self.cachedDistance = None
        self.cachedHubCanServe = None
        return self

    def addNodes(self, nodes: List[HubNode]):
        for node in nodes:
            self.addNode(node)
        return self

    def insertNodeAfter(self, i, newNode: HubNode):
        # insert node after specified index
        #appendAfterIndex = self._findIndexOfID(reqID)
        self.nodes.insert(i+1, newNode)
        self.cachedDistance = None
        self.cachedHubCanServe = None
        return self

    def insertNodeBefore(self, i, newNode: HubNode):
        # insert node after specified index
        #appendBeforeIndex = self._findIndexOfID(reqID)
        self.nodes.insert(i, newNode)
        self.cachedDistance = None
        self.cachedHubCanServe = None
        return self

    def deleteNode(self, node):
        self.nodes.remove(node)
        self.cachedDistance = None
        self.cachedHubCanServe = None

    def deleteNodes(self, nodes: List[HubNode]):
        for node in nodes:
            self.deleteNode(node)
        self.cachedDistance = None
        self.cachedHubCanServe = None

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

    def length(self, distanceMatrix: DistanceMatrix):  # distance length
        if self.cachedDistance:
            return self.cachedDistance

        # distance between nodes (hubs)
        dist = 0
        for i in range(0, len(self.nodes)-1):
            dist += distanceMatrix.byLocID(
                self.nodes[i].locID, self.nodes[i+1].locID)

        # distance from depot to first hub and from last hub to depot

        dist += distanceMatrix.byLocID(
            self.depotLocID, self.nodes[0].locID)
        dist += distanceMatrix.byLocID(
            self.depotLocID, self.nodes[-1].locID)

        #self.cachedDistance = dist
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

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return f"DepotRoute: " + " -> ".join([_.__repr__() for _ in self.nodes])

    def __len__(self):  # nodes (except hub)
        return len(self.nodes)

    def __getitem__(self, i):
        return self.nodes[i]


class DayDepotRoutes(object):
    def __init__(self):
        self.routes = []

    def addRoute(self, route: DepotRoute):
        self.routes.append(route)

    def addRouteFromNodes(self, routeID, hubLocID, nodes: List[HubNode]):
        route = DepotRoute(routeID, hubLocID).addNodes(nodes)
        self.routes.append(route)
        # nodes: [nodeID1, nodeID2, nodeID3, nodeID4]

    def computeCost(self, truckDistanceCost, truckDayCost, distanceMatrix: DistanceMatrix, instance=None):
        # compute cost of distance and #vans
        cost = 0
        for route in self.routes:
            cost += route.length(distanceMatrix) * truckDistanceCost
            cost += truckDayCost

        return cost

    def isFeasible(self, truckCapacity, truckMaxDistance, distanceMatrix, instance, verbose=False) -> bool:
        # check max capacity and maxd istance constraints. None of the operators will remove nodes from the routes so it is assumed that
        # if all requests are served in the initial solution this will be the case for all derived solutions.
        notOverCapacity = True
        notOverDistance = True
        hubCanServeRequests = True
        for route in self.routes:
            if route.demand() > truckCapacity:
                if verbose:
                    print(
                        f"Route with id {route.routeID} has a demand of {route.demand()} > truckCapacity = {truckCapacity}")
                notOverCapacity = False
            routeLength = route.length(distanceMatrix)

            if routeLength > truckMaxDistance:
                if verbose:
                    print(
                        f"Route with id {route.routeID} has a length of {routeLength} > truckMaxDistance = {truckMaxDistance}")
                notOverDistance = False
        feasible = notOverCapacity and notOverDistance and hubCanServeRequests
        return feasible

    def __repr__(self):
        res = ""
        for route in self.routes:
            res += route.__repr__() + "\n"
        return res

    def __len__(self):
        return len(self.routes)


class DepotRoutes(object):
    def __init__(self):
        self.depotRoutes = {}
        self.costCaching = {}

    def addDayDepotRoutes(self, day, dayDepotRoutes: DayDepotRoutes):
        self.depotRoutes[day] = dayDepotRoutes
        self.costCaching[day] = None

    def isFeasible(self, truckCapacity, truckMaxDistance, dm, instance, verbose=False):
        for day, dayDepotRoutes in self.depotRoutes.items():
            if not dayDepotRoutes.isFeasible(truckCapacity, truckMaxDistance, dm, instance, verbose):
                return False
        return True

    def computeCost(self, truckDistanceCost, truckDayCost, truckCost, distanceMatrix: DistanceMatrix, instance=None):
        cost = 0

        for day, dayDepotRoutes in self.depotRoutes.items():
            if self.costCaching[day]:
                cost += self.costCaching[day]
            else:
                dayCost = dayDepotRoutes.computeCost(
                    truckDistanceCost, truckDayCost, distanceMatrix)
                cost += dayCost
                #self.costCaching[day] = dayCost #turned off caching for now

        cost += self.nTrucksRequired() * truckCost
        return cost

    def nTrucksRequired(self):
        return max([len(dayDepotRoutes) for day, dayDepotRoutes in self.depotRoutes.items()])

    def days(self):
        return list(self.depotRoutes.keys())

    def toDict(self, instance):
        res = {}
        depot = instance.Locations[0]
        depotNode = {'X': depot.X, 'Y': depot.Y, 'locID': depot.ID}
        for day, dayDepotRoutes in self.depotRoutes.items():
            dayres = {}
            i = 0
            for route in dayDepotRoutes.routes:
                dayres[i] = [depotNode] + [{'X': n.X, 'Y': n.Y, 'demand': n.demand,'amounts': n.amounts, 'locID': n.locID} for n in route.nodes] + [depotNode]
                i += 1
            res[day] = dayres
        return res


    def __repr__(self):
        res = ""
        for day, data in self.depotRoutes.items():
            res += "-"*10 + f"day {day}" + "-"*10 + "\n"
            res += data.__repr__() + "\n"
        return res

    def __getitem__(self, i):
        return self.depotRoutes[i]

    def copy(self):
        return copy.deepcopy(self)


class Solution(object):
    def __init__(self, hubRoutes: HubRoutes, depotRoutes: DepotRoutes):
        self.depotRoutes = depotRoutes
        self.hubRoutes = hubRoutes

        # this will keep track of which days a truckroute recompute is necessary (after operator is applied to hub on that day)
        self.depotRecomputeNecessary = {day: False for day in hubRoutes.days()}

    def computeCost(self, instance: InstanceCO22, hubCost=True, depotCost=True, distanceMatrix=None, useHubOpeningCost=None):
        # compute solution cost
        cost = 0
        if hubCost:
            cost += self.hubRoutes.computeCost(
                instance.VanDistanceCost, vanDayCost=instance.VanDayCost, vanCost=instance.VanCost, deliverEarlyPenalty=instance.deliverEarlyPenalty, distanceMatrix=distanceMatrix,
                useHubOpeningCost=useHubOpeningCost, instance=instance)

        if depotCost:
            cost += self.depotRoutes.computeCost(
                truckDistanceCost=instance.TruckDistanceCost, truckDayCost=instance.TruckDayCost, truckCost=instance.TruckCost, distanceMatrix=distanceMatrix, instance=instance)
               

        return cost

    def isFeasible(self, instance: InstanceCO22, distanceMatrix, useCheckHubCanServe=None, hub=True, depot=True):
        hubFeasible = True
        depotFeasible = True
        if hub:
            hubFeasible = self.hubRoutes.isFeasible(
                VanCapacity=instance.VanCapacity, VanMaxDistance=instance.VanMaxDistance, dm=distanceMatrix, instance=instance, useCheckHubCanServe=useCheckHubCanServe)
            
        if depot:
            depotFeasible = self.depotRoutes.isFeasible(
                truckCapacity=instance.TruckCapacity, truckMaxDistance=instance.TruckMaxDistance, dm=distanceMatrix, instance=instance)
        solutionFeasible = hubFeasible and depotFeasible
        return solutionFeasible

    def recomputeNecessary(self):
        return any([v for d, v in self.depotRecomputeNecessary.items()])

    def computeDepotSolution(self, instance: InstanceCO22, useDMin = True):
        # recompute depot solution when infeasible because hubsolution has been changed
        redo = [d for d, v in self.depotRecomputeNecessary.items() if v is True]
        if len(redo) > 0:
            self.depotRoutes = initialSolutions.solveDepotDCCached(instance, self.hubRoutes, cache=self.depotRoutes, redo=redo, useDMin=useDMin)
        
        for day in redo:    #reset costCaching of depotRoutes for recomputed days
            self.depotRoutes.costCaching[day] = None

        self.depotRecomputeNecessary = {day: False for day, _ in self.depotRecomputeNecessary.items()} #reset all

    def copy(self):
        return copy.deepcopy(self)

    def toDict(self, instance):
        hubRoutesDict = self.hubRoutes.toDict(instance)
        depotRoutesDict = self.depotRoutes.toDict(instance)
        return {'depotRoutes': depotRoutesDict, 'hubRoutes': hubRoutesDict}

    def toStr(self, instance):
        solutionDict = self.toDict(instance)
        solutionStr = self.solutionToStr(instance, solutionDict)
        return solutionStr

    ## HubRoute operators
    # intraday operators
    def hubIntraDayRandomMergeRoutes(self, day):
        self.hubRoutes.intraDayRandomMergeRoutes(day)
        self.depotRecomputeNecessary[day] = True

    def hubIntraDayRandomNodeInsertion(self, day):
        self.hubRoutes.intraDayRandomNodeInsertion(day)
        self.depotRecomputeNecessary[day] = True

    def hubIntraDayRandomSectionInsertion(self, day):
        self.hubRoutes.intraDayRandomSectionInsertion(day)
        self.depotRecomputeNecessary[day] = True
        
    def hubIntraDayShuffleRoute(self, day):
        self.hubRoutes.intraDayShuffleRoute(day) 
        
    # Multiday operators
    def hubRandomMoveNodeDayEarly(self):
        day = random.randint(2, 20)
        while len(self.hubRoutes.hubRoutes[day]) < 1:
            day = random.randint(2, 20)

        self.hubRoutes.randomMoveNodeDayEarly(day=day)
        self.depotRecomputeNecessary[day-1] = True
        self.depotRecomputeNecessary[day-1] = True


    def hubRandomChooseOtherHub(self, allHubs: set):
        self.hubRoutes.randomChooseOtherHub(allHubs=allHubs)
        self.depotRecomputeNecessary = {day: True for day, _ in self.depotRecomputeNecessary.items()} #reset all


    ##Write solutiondict to solution string format

    def solutionToStr(self, instance: InstanceCO22, res: dict):
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
