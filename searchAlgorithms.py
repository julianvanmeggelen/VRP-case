from dataClasses import *
from typing import Callable
from validator.InstanceCO22 import InstanceCO22
import time
import random
import numpy as np




class randomLocalSearch(object):
    def __init__(self, instance: InstanceCO22, maxIterations: int, maxTimeSeconds: int = math.inf, nStart:int = 1, initialState: HubRoutes=None, initialStateGenerator: Callable[[InstanceCO22], HubRoutes] = None):

        if initialStateGenerator:
            self.initialState = initialStateGenerator(instance)
            print("Generated initial state")
        else:
            self.initialState = initialState

        self.instance = instance
        self.maxIterations = maxIterations/nStart
        self.maxTimeSeconds = maxTimeSeconds/nStart
        self.nStart = nStart
        self.allHubLocIDs = set([_.ID+1 for _ in self.instance.Hubs])
        self.useHubOpeningCost = any([_.hubOpeningCost > 0 for _ in instance.Hubs])
        self.distanceMatrix = DistanceMatrix(instance)
        self.optimizeDeliverEarly = (instance.deliverEarlyPenalty > 0)
        self.useCheckHubCanServe = any([len(_.allowedRequests) != len(instance.Requests) for _ in instance.Hubs])
        print("useCheckHubCanServe", self.useCheckHubCanServe)
        self.currentState = self.initialState
        self.useMoveDayEarly = instance.deliverEarlyPenalty > 0
        self.currentStateCost = self.computeStateCost(self.initialState)
        self.initialStateCost = self.currentStateCost.copy()
        self.bestState = self.initialState
        self.bestStateCost = self.initialStateCost
        self.costs = [self.initialStateCost]

        self.N_OPERATORS = 4
        self.N_DAYS = instance.Days

        #pregenerate operators to apply
        self.operatorIDs = list(np.random.randint(0,self.N_OPERATORS, maxIterations*self.N_DAYS+1))

    def computeStateCost(self, state):
        return state.computeCost(self.instance.VanDistanceCost, self.instance.VanDayCost, self.instance.VanCost, self.instance.deliverEarlyPenalty, self.distanceMatrix, self.useHubOpeningCost, self.instance)

    def checkFeasibleState(self, state):
        return state.isFeasible(self.instance.VanCapacity, self.instance.VanMaxDistance, self.distanceMatrix, self.instance, verbose=False, useCheckHubCanServe=self.useCheckHubCanServe)

    def checkBetterState(self, neighbour, i, operator):
        neighbourCost = self.computeStateCost(neighbour)
        if neighbourCost < self.currentStateCost:
            if self.checkFeasibleState(neighbour):
                self.currentState = neighbour
                self.currentStateCost = neighbourCost
                self.costs.append(self.currentStateCost)
                if self.verbose:
                    self._progressBar(i, f"[{i}] NB better than current state: {neighbourCost} {operator}")
                    return
        self._progressBar(i, msg="")
        return
        


    def searchNeighbour(self, i):

        if self.useHubOpeningCost:
            neighbour = self.currentState.copy()
            neighbour.randomChooseOtherHub(allHubs = self.allHubLocIDs)
            self.checkBetterState(neighbour, i, "HubRoutesOperator randomChooseOtherHub")

        for day in range(1,self.instance.Days+1): # apply operator to every day and check if better
            operatorID = self.operatorIDs.pop(0)
            neighbour = self.currentState.copy()
            neighbour.applyOperator(day, operatorID)
            self.checkBetterState(neighbour, i, f"DayHubRoutes operator {operatorID}")

        if self.useMoveDayEarly:
            neighbour = self.currentState.copy()
            neighbour.randomMoveNodeDayEarly()
            self.checkBetterState(neighbour, i, "HubRoutesOperator randomMoveNodeDayEarly")


    def _progressBar(self, current, msg, bar_length=20):
        fraction = current / self.maxIterations
        arrow = int(fraction * bar_length - 1) * '-' + '>'
        padding = int(bar_length - len(arrow)) * ' '

        ending = '\n' if current == self.maxIterations else '\r'
        prct = f"{int(fraction*100)}%"

        print(f'Progress: [{arrow}{padding}] {prct.ljust(4)} {self.currentStateCost} {msg}', end=ending)

    def run(self, verbose=True):
        self.verbose = verbose
        for iteration in range(self.nStart):
            startTime = time.time()

            if self.verbose:
                print("-"*20, f'Run {iteration}', "-"*20)
                print(f"initialCost: {self.initialStateCost}")
            i = 0

            while time.time() - startTime < self.maxTimeSeconds and i < self.maxIterations:
                self.searchNeighbour(i)
                i+=1
                

            print(f"\n End of iteration, stateCost obtained: {self.currentStateCost}")
            
            if self.currentStateCost < self.bestStateCost:
                self.bestStateCost = self.currentStateCost
                self.bestState = self.currentState
                if self.verbose:
                    print(f"Current better than best state: {self.currentStateCost}")
               
            self.currentState = self.initialState # reset to initial state after this iteration ended
            self.currentStateCost = self.initialStateCost

        return self.bestState, self.costs



##
#   EXPERIMENT Tree traversal search
##

class TreeSearch(object):
    def __init__(self, instance: InstanceCO22, initialState: HubRoutes, infeasibilityLevel = 0, searchLevel = 0, maxChildren = 10, initialStateCost = None, parent = None):
        self.initialState = initialState
        self.infeasibilityLevel = infeasibilityLevel
        self.searchLevel = searchLevel
        self.maxChildren = maxChildren
        self.bestState = initialState

        if parent:
            self.allHubLocIDs = parent.allHubLocIDs
        else:
            self.allHubLocIDs = set([_.ID+1 for _ in self.instance.Hubs])

        if initialStateCost:
            self.bestStateCost = initialStateCost
        else:
            self.bestStateCost = self.computeStateCost(initialState)

        self.N_OPERATORS = 6
        self.MAX_INFEASIBILITY_LEVEL = 5


    def computeStateCost(self, state):
        return state.computeCost(self.instance.VanDistanceCost, self.instance.VanDayCost, self.instance.VanCost, self.instance.deliverEarlyPenalty, self.distanceMatrix, self.useHubOpeningCost, self.instance)

    def checkFeasibleState(self, state):
        return state.isFeasible(self.instance.VanCapacity, self.instance.VanMaxDistance, self.distanceMatrix, self.instance, verbose=False, useCheckHubCanServe=self.useCheckhHubCanServe)

    def checkBetterState(self, neighbour, i, operator):
        neighbourCost = self.computeStateCost(neighbour)
        if neighbourCost < self.currentStateCost:
            if self.checkFeasibleState(neighbour):
                self.bestStae = neighbour
                self.bestStateCost = neighbourCost
                if self.verbose:
                    #self._progressBar(i, f"[{i}] NB better than current state: {neighbourCost} {operator}")
                    return
        #self._progressBar(i, msg="")
        return

    def generateChild(self, randomNr):
        neighbour = self.initialState.copy()
        if randomNr ==0:
            neighbour.randomMergeRoutes()
        elif randomNr ==1:
            neighbour.randomNodeInsertion()
        elif randomNr ==2:
            neighbour.randomSectionInsertion()
        elif randomNr ==3:
            neighbour.randomMoveNodeDayEarly()
        elif randomNr ==4:  
            neighbour.randomChooseOtherHub(self.allHubLocIDs)
        
        return neighbour


    def generateChildren(self) -> List[HubRoutes]:
        self.neighbours = []
        operators = list(np.random.randint(0,self.N_OPERATORS, self.maxChildren))   
        for i in range(self.maxChildren):
            neighbour = self.generateChild(operators.pop())
            self.neighbours.append(neighbour)

    def searchChildren(self):
        childBestStates = []
        for child in self.neighbours:
            neighBourCost = self.computeStateCost(child)
            if neighBourCost < self.bestStateCost:
                if self.checkFeasibleState(child):
                    newSearcher = TreeSearch(child, infeasibilityLevel=0,searchLevel=self.searchLevel+1, maxChildren=self.maxChildren)
                    newSearcherBestState = newSearcher.search()
                    childBestStates
                else:
                    if self.infeasibilityLevel + 1 > self.MAX_INFEASIBILITY_LEVEL:
                        return 
                    else:
                        newSearcher = TreeSearch(child, infeasibilityLevel=self.infeasibilityLevel+1,searchLevel=self.searchLevel+1, maxChildren=self.maxChildren)
                        newSearcherBestState = newSearcher.search()


    def search(self) -> HubRoutes:
        return 





    

