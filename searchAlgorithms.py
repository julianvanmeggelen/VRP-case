from pyclbr import Function
from xmlrpc.client import Boolean, boolean
from dataClasses import *
from typing import Callable
from validator.InstanceCO22 import InstanceCO22
import time
import random
import numpy as np
from joblib import Parallel, delayed, cpu_count




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

        self.instance = instance
        self.initialState = initialState
        self.infeasibilityLevel = infeasibilityLevel
        self.searchLevel = searchLevel
        self.maxChildren = maxChildren
        self.bestState = initialState

        if parent:  #if parent is provided no need to recompute these properties
            self.allHubLocIDs = parent.allHubLocIDs
            self.useHubOpeningCost = parent.useHubOpeningCost
            self.distanceMatrix = parent.distanceMatrix
            self.useCheckHubCanServe = parent.useCheckHubCanServe
        else:
            self.allHubLocIDs = set([_.ID+1 for _ in self.instance.Hubs])
            self.useHubOpeningCost = any([_.hubOpeningCost > 0 for _ in instance.Hubs])
            self.distanceMatrix = DistanceMatrix(instance)
            self.useCheckHubCanServe = any([len(_.allowedRequests) != len(instance.Requests) for _ in instance.Hubs])

        if initialStateCost:
            self.initialStateCost = initialStateCost
        else:
            self.initialStateCost = self.computeStateCost(initialState)

        self.N_OPERATORS = 5
        self.MAX_INFEASIBILITY_LEVEL = 0


    def computeStateCost(self, state):
        return state.computeCost(self.instance.VanDistanceCost, self.instance.VanDayCost, self.instance.VanCost, self.instance.deliverEarlyPenalty, self.distanceMatrix, self.useHubOpeningCost, self.instance)

    def checkFeasibleState(self, state):
        return state.isFeasible(self.instance.VanCapacity, self.instance.VanMaxDistance, self.distanceMatrix, self.instance, verbose=False, useCheckHubCanServe=self.useCheckHubCanServe)

    def generateChild(self, randomNr, day):
        #using the random assigned operator, generate one child

        #self._log(f"Applied operator {randomNr} to day {day}")
        neighbour = self.initialState.copy()
        if randomNr ==0:
            neighbour.applyOperator(day, 0)
        elif randomNr ==1:
            neighbour.applyOperator(day, 1)
        elif randomNr ==2:
            neighbour.applyOperator(day, 2)
        elif randomNr ==3:
            neighbour.applyOperator(day, 3)
        elif randomNr ==4:  
            neighbour.randomChooseOtherHub(self.allHubLocIDs)
        elif randomNr ==5:
            neighbour.randomMoveNodeDayEarly()
        
        return neighbour

    def generateChildren(self) -> List[HubRoutes]:
        #generate maxChildren child nodes
        self.neighbours = []
        operators = list(np.random.randint(0,self.N_OPERATORS+1, self.maxChildren))   
        days = list(np.random.randint(1, 21, self.maxChildren))   
        for i in range(self.maxChildren):
            neighbour = self.generateChild(operators.pop(), days.pop())
            self.neighbours.append(neighbour)

    def _log(self, msg):
        spacer = " " * self.searchLevel + "|- "
        print(spacer + msg)


    def searchChildren(self):
        #evaluate cost of every child (neighbour), if cost is better than current and state is feasible, start search from the child, if cost is better but not feasible start new search with 1 deducted from infeasibilityLevel
        self.childBestState = self.initialState
        self.childBestStateCost = self.initialStateCost



        for child in self.neighbours:
            #self._log("Searching new child")
            neighbourCost = self.computeStateCost(child)
            if neighbourCost < self.initialStateCost:
                self._log(f"Child has better cost: {neighbourCost}")

                if self.checkFeasibleState(child):  #child yields better cost and is feasible
                    newSearcher = TreeSearch(self.instance, child, infeasibilityLevel=0,searchLevel=self.searchLevel+1, maxChildren=self.maxChildren, initialStateCost = neighbourCost, parent=self)
                    newSearcherBestState, newSearcherBestStateCost = newSearcher.run()
                else:                               #child yields better cost but is not feasible
                    if self.infeasibilityLevel + 1 > self.MAX_INFEASIBILITY_LEVEL:
                        self._log(f" {self.searchLevel} Branch exited because max infeasibilitylevel reached")
                        continue                    #reached max infeasibility level
                    else:
                        newSearcher = TreeSearch(self.instance, child, infeasibilityLevel=self.infeasibilityLevel+1,searchLevel=self.searchLevel+1, maxChildren=self.maxChildren, initialStateCost = neighbourCost, parent=self)
                        newSearcherBestState, newSearcherBestStateCost = newSearcher.run()
                
                #check if the child is better than the current best child
                if newSearcherBestStateCost < self.childBestStateCost and self.checkFeasibleState(newSearcherBestState):
                    self.childBestState = newSearcherBestState
                    self.childBestStateCost = newSearcherBestStateCost
            else:
                continue
                #self._log("Child not better cost")
        
        return self.childBestState, self.childBestStateCost

    def run(self) -> HubRoutes:
        spacer = (self.searchLevel-1)*" " + "_"
        print(spacer)
        self._log(f"Starting search on branch with infeasibility level {self.infeasibilityLevel} and search level {self.searchLevel} and initialCost {self.initialStateCost}")
        self.generateChildren()
        childBestState, childBestStateCost = self.searchChildren()
        self._log(f"Finished search on branch starting at initialCost {self.initialStateCost} finishing at {childBestStateCost}")
        return childBestState, childBestStateCost

class EvolutionarySearchOld(object):
    def __init__(self, instance: InstanceCO22, initialState: HubRoutes, infeasibilityLevel = 0, searchLevel = 0, maxChildren = 10, initialStateCost = None, parent = None):

        self.instance = instance
        self.initialState = initialState
        self.infeasibilityLevel = infeasibilityLevel
        self.searchLevel = searchLevel
        self.maxChildren = maxChildren
        self.bestState = initialState

        if parent:  #if parent is provided no need to recompute these properties
            self.allHubLocIDs = parent.allHubLocIDs
            self.useHubOpeningCost = parent.useHubOpeningCost
            self.distanceMatrix = parent.distanceMatrix
            self.useCheckHubCanServe = parent.useCheckHubCanServe
        else:
            self.allHubLocIDs = set([_.ID+1 for _ in self.instance.Hubs])
            self.useHubOpeningCost = any([_.hubOpeningCost > 0 for _ in instance.Hubs])
            self.distanceMatrix = DistanceMatrix(instance)
            self.useCheckHubCanServe = any([len(_.allowedRequests) != len(instance.Requests) for _ in instance.Hubs])

        if initialStateCost:
            self.initialStateCost = initialStateCost
        else:
            self.initialStateCost = self.computeStateCost(initialState)

        self.N_OPERATORS = 5
        self.MAX_INFEASIBILITY_LEVEL = 0
        self.SELECTION_SIZE = 1


    def computeStateCost(self, state):
        return state.computeCost(self.instance.VanDistanceCost, self.instance.VanDayCost, self.instance.VanCost, self.instance.deliverEarlyPenalty, self.distanceMatrix, self.useHubOpeningCost, self.instance)

    def checkFeasibleState(self, state):
        return state.isFeasible(self.instance.VanCapacity, self.instance.VanMaxDistance, self.distanceMatrix, self.instance, verbose=False, useCheckHubCanServe=self.useCheckHubCanServe)

    def generateChild(self, randomNr, day):
        #using the random assigned operator, generate one child

        #self._log(f"Applied operator {randomNr} to day {day}")
        neighbour = self.initialState.copy()
        if randomNr ==0:
            neighbour.applyOperator(day, 0)
        elif randomNr ==1:
            neighbour.applyOperator(day, 1)
        elif randomNr ==2:
            neighbour.applyOperator(day, 2)
        elif randomNr ==3:
            neighbour.applyOperator(day, 3)
        elif randomNr ==4:  
            neighbour.randomChooseOtherHub(self.allHubLocIDs)
        elif randomNr ==5:
            neighbour.randomMoveNodeDayEarly()
        
        return neighbour

    def generateChildren(self) -> List[HubRoutes]:
        #generate maxChildren child nodes
        self.neighbours = []
        operators = list(np.random.randint(0,self.N_OPERATORS+1, self.maxChildren))   
        days = list(np.random.randint(1, 21, self.maxChildren))   
        for i in range(self.maxChildren):
            neighbour = self.generateChild(operators.pop(), days.pop())
            self.neighbours.append(neighbour)

    def _log(self, msg):
        spacer = " " * self.searchLevel + "|-"
        print(spacer, msg)


    def searchChildren(self):
        #evaluate cost of every child (neighbour), if cost is better than current and state is feasible, start search from the child, if cost is better but not feasible start new search with 1 deducted from infeasibilityLevel
        self.childBestState = self.initialState
        self.childBestStateCost = self.initialStateCost

        neighbourFeasible = [self.checkFeasibleState(nb) for nb in self.neighbours]
        neighbourCosts = [self.computeStateCost(nb) if neighbourFeasible[i] else math.inf for i, nb in enumerate(self.neighbours)]

        selectedNeighboursIndices = sorted(range(len(neighbourCosts)), key=lambda k: neighbourCosts[k])[0:self.SELECTION_SIZE]
        selectedNeighbours = [_ for i,_ in enumerate(self.neighbours) if i in selectedNeighboursIndices]

        for i, child in enumerate(selectedNeighbours):
            #self._log("Searching new child")
            neighbourCost = neighbourCosts[i]

            if neighbourFeasible[i]: #check feasible 
                newSearcher = EvolutionarySearchOld(self.instance, child, infeasibilityLevel=self.infeasibilityLevel+1,searchLevel=self.searchLevel+1, maxChildren=self.maxChildren, initialStateCost = neighbourCost, parent=self)
                newSearcherBestState, newSearcherBestStateCost = newSearcher.run()
                    
                #check if the child is better than the current best child
                if newSearcherBestStateCost < self.childBestStateCost:
                    self.childBestState = newSearcherBestState
                    self.childBestStateCost = newSearcherBestStateCost
                else:
                    continue
                    #self._log("Child not better cost")
        
        return self.childBestState, self.childBestStateCost

    def run(self) -> HubRoutes:
        spacer = (self.searchLevel-1)*" " + "_"
        print(spacer)
        self._log(f"Starting search on branch with infeasibility level {self.infeasibilityLevel} and search level {self.searchLevel} and initialCost {self.initialStateCost}")
        self.generateChildren()
        childBestState, childBestStateCost = self.searchChildren()
        self._log(f"Finished search on branch starting at initialCost {self.initialStateCost} finishing at {childBestStateCost}")
        return childBestState, childBestStateCost

class EvolutionarySearch(object):
    #evolutionary search on HubRoutes
    def __init__(self, instance: InstanceCO22, initialState: HubRoutes, generationSize, candidateSize, nGenerations):


        self.instance = instance
        self.distanceMatrix = DistanceMatrix(instance)
        self.initialState = initialState

        self.allHubLocIDs = set([_.ID+1 for _ in self.instance.Hubs])
        self.useHubOpeningCost = any([_.hubOpeningCost > 0 for _ in instance.Hubs])
        self.useCheckHubCanServe = any([len(_.allowedRequests) != len(instance.Requests) for _ in instance.Hubs])
        self.optimizeDeliverEarly = (instance.deliverEarlyPenalty > 0)

        self.initialStateCost = self.computeStateCost(initialState)
        self.bestState = initialState
        self.bestStateCost = self.initialStateCost
        self.costs = [self.bestStateCost]
        self.allCosts = [[self.bestStateCost]]


        self.generations = [[initialState]]

        self.N_OPERATORS = 4 + self.optimizeDeliverEarly

        self.generationSize, self.candidateSize, self.nGenerations = generationSize, candidateSize, nGenerations


    def computeStateCost(self, state):
        return state.computeCost(self.instance.VanDistanceCost, self.instance.VanDayCost, self.instance.VanCost, self.instance.deliverEarlyPenalty, self.distanceMatrix, self.useHubOpeningCost, self.instance)

    def checkFeasibleState(self, state):
        return state.isFeasible(self.instance.VanCapacity, self.instance.VanMaxDistance, self.distanceMatrix, self.instance, verbose=False, useCheckHubCanServe=self.useCheckHubCanServe)

    def generateChild(self, mem, randomNr=None, day=None):
        #using the random assigned operator, generate one child

        if randomNr == None:
            randomNr = random.randint(0,self.N_OPERATORS)

        if day == None:
            day = random.randint(1,20)

        #self._log(f"Applied operator {randomNr} to day {day}")

        neighbour = mem.copy()
        if randomNr ==0:
            neighbour.intraDayRandomMergeRoutes(day)
        elif randomNr ==1:
            neighbour.intraDayRandomNodeInsertion(day)
        elif randomNr ==2:
            neighbour.intraDayRandomSectionInsertion(day)
        elif randomNr ==3:
            neighbour.intraDayShuffleRoute(day)
        elif randomNr ==4:  
            neighbour.randomChooseOtherHub(self.allHubLocIDs)
        elif randomNr ==5:
            neighbour.randomMoveNodeDayEarly()
        
        return neighbour

    def generateCandidates(self, prevGen: List[HubRoutes]) -> List[HubRoutes]:

        nextGen = []
        prevGenSize = len(prevGen)
        candidatesPerGenMember = round(self.candidateSize / prevGenSize)

        operators = list(np.random.randint(0,self.N_OPERATORS+1, self.candidateSize ))   
        days = list(np.random.randint(1, 21, self.candidateSize ))

        for mem in prevGen:
            for i in range(candidatesPerGenMember):
                neighbour = self.generateChild(mem, operators.pop(), days.pop())
                nextGen.append(neighbour)

        return nextGen

    def generateCandidatesParallel(self, prevGen: List[HubRoutes]) -> List[HubRoutes]:

        def generateBatch(batch):
            #generate candidatesPerGenMember neighbours for mem
            res = []
            for mem in batch:
                for i in range(candidatesPerGenMember):
                    res.append(self.generateChild(mem))
            return res
    
        nextGen = []
        batches = np.array_split(prevGen, self.nJobs)
        prevGenSize = len(prevGen)
        candidatesPerGenMember = round(self.candidateSize / prevGenSize)

        operators = list(np.random.randint(0,self.N_OPERATORS+1, self.candidateSize ))   
        days = list(np.random.randint(1, 21, self.candidateSize ))

        
        res = Parallel(n_jobs=-1)(delayed(generateBatch)(batch) for batch in batches)

        for batch in res:
            nextGen += batch

        return nextGen

    def selectNextGeneration(self, candidates: List[HubRoutes]) -> List[HubRoutes]: #returns sorted nextgeneration
        neighbourFeasible = [self.checkFeasibleState(nb) for nb in candidates]
        neighbourCosts = [self.computeStateCost(nb) if neighbourFeasible[i] else math.inf for i, nb in enumerate(candidates)]
        #print(neighbourCosts)

        selectedNeighboursIndices = sorted(range(len(neighbourCosts)), key=lambda k: neighbourCosts[k])[0:self.generationSize]
        selectedNeighboursCosts = [neighbourCosts[i] for i in selectedNeighboursIndices]

        self.allCosts.append(selectedNeighboursCosts)

        selectedNeighbours = [_ for i,_ in enumerate(candidates) if i in selectedNeighboursIndices and neighbourFeasible[i]]
        #print(neighbourCosts[selectedNeighboursIndices[0]])
        if neighbourCosts[selectedNeighboursIndices[0]] < self.bestStateCost: # check if best of generation is better than current best
            self.bestState = selectedNeighbours[0]
            self.bestStateCost = neighbourCosts[selectedNeighboursIndices[0]]

        return selectedNeighbours

    def run(self, parallel = True, earlyStopping: Callable[[List[float]],bool] = lambda costs: False):

        if parallel:
            self.nJobs = cpu_count()
            print(f"Using parallel computing for neighbour generation with nJobs = {self.nJobs}")
        startTime=time.time()
        for i in range(1, self.nGenerations+2):
            iterationStartTime = time.time()

            prevGen = self.generations[i-1]
            if parallel:
                nextGenCandidates = self.generateCandidatesParallel(prevGen)
            else:
                nextGenCandidates = self.generateCandidates(prevGen)
            nextGen = self.selectNextGeneration(nextGenCandidates)
            self.generations.append(nextGen)
            self.costs.append(self.bestStateCost)

            elapsedTime = time.time() - iterationStartTime

            if earlyStopping(self.costs):
                break

            print(f"Generation {i} - Size: {len(self.generations[-1])} bestCost: {self.bestStateCost} generationCostVariance: {np.std(self.allCosts[-1]):.2f} elapsed: {elapsedTime:.2f}")
        elapsedTime = time.time() - startTime
        print(f"Finished in {elapsedTime:.2f} resulting in cost of: {self.initialStateCost} -> {self.bestStateCost}")
        return self.bestState, self.bestStateCost

class EvolutionarySearchBothEchelons(object):
    #evolutionarySearch on Solution

    def __init__(self, instance: InstanceCO22, initialState: Solution, generationSize, candidateSize, nGenerations=50):


        self.instance = instance
        self.distanceMatrix = DistanceMatrix(instance)
        self.initialState = initialState

        self.allHubLocIDs = set([_.ID+1 for _ in self.instance.Hubs])
        self.useHubOpeningCost = any([_.hubOpeningCost > 0 for _ in instance.Hubs])
        self.useCheckHubCanServe = any([len(_.allowedRequests) != len(instance.Requests) for _ in instance.Hubs])
        self.optimizeDeliverEarly = (instance.deliverEarlyPenalty > 0)

        
        self.bestState = initialState
        self.costsInitialised = False
        self.generations = [[initialState]]

        self.N_OPERATORS = 4 + self.optimizeDeliverEarly

        self.generationSize, self.candidateSize, self.nGenerations = generationSize, candidateSize, nGenerations


    def computeStateCost(self, state):
        return state.computeCost(instance=self.instance, hubCost=True, depotCost=self.useDepotCost, distanceMatrix=self.distanceMatrix, useHubOpeningCost=self.useHubOpeningCost)

    def checkFeasibleState(self, state, depot=True):
        return state.isFeasible(instance=self.instance, distanceMatrix=self.distanceMatrix, useCheckHubCanServe=self.useCheckHubCanServe, depot=depot)

    def generateChild(self, mem, randomNr=None, day=None):
        #using the random assigned operator, generate one child

        if randomNr == None:
            randomNr = random.randint(0,self.N_OPERATORS)

        if day == None:
            day = random.randint(1,20)

        #self._log(f"Applied operator {randomNr} to day {day}")

        neighbour = mem.copy()
        if randomNr ==0:
            neighbour.hubIntraDayRandomMergeRoutes(day)
        elif randomNr ==1:
            neighbour.hubIntraDayRandomNodeInsertion(day)
        elif randomNr ==2:
            neighbour.hubIntraDayRandomSectionInsertion(day)
        elif randomNr ==3:
            neighbour.hubIntraDayShuffleRoute(day)
        elif randomNr ==4:  
            neighbour.hubRandomChooseOtherHub(self.allHubLocIDs)
        elif randomNr ==5:
            neighbour.hubRandomMoveNodeDayEarly()
        
        return neighbour

    def generateCandidates(self, prevGen: List[Solution]) -> List[Solution]:

        nextGen = []
        prevGenSize = len(prevGen)
        candidatesPerGenMember = round(self.candidateSize / prevGenSize)

        operators = list(np.random.randint(0,self.N_OPERATORS+1, self.candidateSize ))   
        days = list(np.random.randint(1, 21, self.candidateSize ))

        for mem in prevGen:
            for i in range(candidatesPerGenMember):
                neighbour = self.generateChild(mem, operators.pop(), days.pop())
                nextGen.append(neighbour)

        return nextGen

    def generateCandidatesParallel(self, prevGen: List[Solution]) -> List[Solution]:

        def generateBatch(batch):
            #generate candidatesPerGenMember neighbours for mem
            res = []
            for mem in batch:
                for i in range(candidatesPerGenMember):
                    res.append(self.generateChild(mem))
            return res
    
        nextGen = []
        batches = np.array_split(prevGen, self.nJobs)
        prevGenSize = len(prevGen)
        candidatesPerGenMember = round(self.candidateSize / prevGenSize)

        operators = list(np.random.randint(0,self.N_OPERATORS+1, self.candidateSize ))   
        days = list(np.random.randint(1, 21, self.candidateSize ))

        
        res = Parallel(n_jobs=-1)(delayed(generateBatch)(batch) for batch in batches)

        for batch in res:
            nextGen += batch

        return nextGen

    def recomputeDepotRoutes(self, feasibleNeighbours):
        if self.recomputeDepotRoute:
            if self.parallel:
                fun = lambda nb: nb.compcomputeDepotSolution(self.instance)
                res = Parallel(n_jobs=-1)(delayed(fun)(nb) for nb in feasibleNeighbours)
            else:
                for nb in feasibleNeighbours:   #recompute depotroutes if necessary
                    nb.computeDepotSolution(instance=self.instance)
        return feasibleNeighbours


    def selectNextGeneration(self, candidates: List[Solution]) -> List[Solution]: #returns sorted nextgeneration
        feasibleNeighbours = [nb for nb in candidates if self.checkFeasibleState(nb, depot=False)]

        feasibleNeighbours = self.recomputeDepotRoutes(feasibleNeighbours)

        neighbourCosts = [self.computeStateCost(nb) for nb in feasibleNeighbours]
        #print(neighbourCosts)

        selectedNeighboursIndices = sorted(range(len(neighbourCosts)), key=lambda k: neighbourCosts[k])[0:self.generationSize]
        selectedNeighboursCosts = [neighbourCosts[i] for i in selectedNeighboursIndices]

        self.allCosts.append(selectedNeighboursCosts)

        selectedNeighbours = [_ for i,_ in enumerate(feasibleNeighbours) if i in selectedNeighboursIndices]
        
        #print(neighbourCosts[selectedNeighboursIndices[0]])
        if neighbourCosts[selectedNeighboursIndices[0]] < self.bestStateCost: # check if best of generation is better than current best
            self.bestState = selectedNeighbours[0]
            self.bestStateCost = neighbourCosts[selectedNeighboursIndices[0]]

        return selectedNeighbours

    def initCosts(self):
        self.initialStateCost = self.computeStateCost(self.initialState)
        self.bestStateCost = self.initialStateCost
        if not self.costsInitialised:
            self.costs = [self.bestStateCost]
            self.allCosts = [[self.bestStateCost]]
            self.costsInitialised = True
        

    def run(self, parallel = True, earlyStopping: Callable[[List[float]],bool] = lambda costs: False, recomputeDepotRoute = True, useDepotCost = True, nGenerations = None):
        self.recomputeDepotRoute = recomputeDepotRoute
        self.useDepotCost = useDepotCost
        self.parallel = parallel

        self.initCosts()
        
        nGenerations = nGenerations if nGenerations else self.nGenerations
        print(f"Recomputing depot route: {recomputeDepotRoute}, using depot route cost: {useDepotCost}, initialStateCost: {self.initialStateCost}")
        if parallel:
            self.nJobs = cpu_count()
            print(f"Using parallel computing for neighbour generation with nJobs = {self.nJobs}")
        startTime=time.time()
        for i in range(len(self.generations), len(self.generations)+ nGenerations+2):
            iterationStartTime = time.time()

            prevGen = self.generations[i-1]
            if parallel:
                nextGenCandidates = self.generateCandidatesParallel(prevGen)
            else:
                nextGenCandidates = self.generateCandidates(prevGen)
            nextGen = self.selectNextGeneration(nextGenCandidates)
            self.generations.append(nextGen)
            self.costs.append(self.bestStateCost)

            elapsedTime = time.time() - iterationStartTime

            if earlyStopping(self.costs):
                break

            print(f"Generation {i} - Size: {len(self.generations[-1])} bestCost: {self.bestStateCost} generationCostVariance: {np.std(self.allCosts[-1]):.2f} elapsed: {elapsedTime:.2f}")
        elapsedTime = time.time() - startTime
        print(f"Finished in {elapsedTime:.2f} resulting in cost of: {self.initialStateCost} -> {self.bestStateCost}")

        self.bestState.computeDepotSolution(instance=self.instance)
        return self.bestState, self.bestStateCost
    

        










    

