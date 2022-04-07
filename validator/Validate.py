#! /usr/bin/env python

"""

@author: caroline jagtenberg based on ortec verologsolver
"""

import os
import argparse, copy
import xml.etree.ElementTree as ET
from InstanceCO22 import InstanceCO22 as InstanceCO22
import baseParser as base
from collections import OrderedDict
from pprint import pprint as pprint
import sys

class SolutionCO22(base.BaseParser):
    parsertype = 'solution'

    class LANG:
        class TXT:
            dataset = 'DATASET'
            nrTrucksUsed = 'NUMBER_OF_TRUCKS_USED'
            nrTruckDays = 'NUMBER_OF_TRUCK_DAYS'
            truckDistance = 'TRUCK_DISTANCE'
            nrVansUsed = 'NUMBER_OF_VANS_USED'
            nrVanDays = 'NUMBER_OF_VAN_DAYS'
            vanDistance = 'VAN_DISTANCE'
            hubsUsed ='HUBS_USED'
            hubCost ='HUB_COST'
            cumulativeEarlyPenalty = 'CUMULATIVE_EARLY_PENALTY'
            totalCost = 'TOTAL_COST'
            day = 'DAY'
            dayNrTrucks = 'NUMBER_OF_TRUCKS'
            dayNrVans = 'NUMBER_OF_VANS'

            costfields  = { truckDistance:    'TruckDistance',
                            nrTruckDays:      'NrTruckDays',
                            nrTrucksUsed:     'NrTrucksUsed',
                            vanDistance:'VanDistance',
                            nrVanDays: 'NrVanDays',
                            nrVansUsed:'NrVansUsed',
                            hubsUsed:  'HubsUsed',
                            hubCost:  'HubCost',
                            cumulativeEarlyPenalty:  'CumulativeEarlyPenalty',
                            totalCost:        'Cost',
                            }
            dayfields   = { dayNrTrucks: 'NrTruckRoutes',
                            dayNrVans: 'NrVanRoutes',
                            }


    class SolutionCost(object):
        def __init__(self):
            self.TruckDistance = None
            self.NrTruckDays = None
            self.NrTrucksUsed = None
            self.VanDistance = None
            self.NrVanDays = None
            self.NrVansUsed = None
            self.Cost = None
            #self.TruckDistance = None
            #self.VanDistance = None
            self.HubsUsed = None #perhaps keep track of hubs *used* first, translate to hubcosts later.
            self.HubCost = None
            self.CumulativeEarlyPenalty = None


        def __str__(self):
            if not self.TruckDistance or not self.NrTruckDays or not self.NrTrucksUsed or not self.VanDistance or not self.NrVanDays or not self.NrVansUsed or not self.HubCost or not self.Cost:# or not self.CumulativeEarlyPenalty (if not allowed to deliver early, then CumulativeEarlyPenalty does not exist )
                return 'Not every entry that I expected seems filled, but I will give you the values anyway: \n TruckDistance: %r\nNrTruckDays: %r\nNrTrucksUsed: %r\nVanDistance: %r\nNrVanDays: %r\nNrVansUsed: %r\nHubCost: %r\nEarlyPenalties: %r\nCost: %r' % (self.TruckDistance, self.NrTruckDays, self.NrTrucksUsed, self.VanDistance, self.NrVanDays,  self.NrVansUsed, self.HubCost, self.CumulativeEarlyPenalty, self.Cost )
            else:
                 return 'TruckDistance: %d\nNrTruckDays: %d\nNrTrucksUsed: %d\nVanDistance: %d\nNrVanDays: %d\nNrVansUsed: %d\nHubCost: %d\nEarlyPenalties: %d\nCost: %d' % (self.TruckDistance, self.NrTruckDays, self.NrTrucksUsed, self.VanDistance, self.NrVanDays,  self.NrVansUsed, self.HubCost, self.CumulativeEarlyPenalty,self.Cost )

    class TruckRoute(object):
        def __init__(self):
            self.ID = None
            self.Route = []
            self.Amounts = [] #this used to be [[]] but that was wrong
            self.calcDistance = None

        def __str__(self):
            return '%d %s' % (self.ID, ' '.join(str(x) for x in self.Route))


    class VanRoute(object):
        def __init__(self):
            self.ID = None
            self.HubID = None
            self.Route = []
            self.calcDistance = None

        def __str__(self):
            result = f'Vanroute with ID={self.ID}, start and end at hub {self.HubID}, requests in route = '
            for x in self.Route:
                result += str(x)
                result += ','
            return result

    class SolutionDay(object):
        def __init__(self, dayNr):
            self.dayNumber = dayNr
            self.NrTruckRoutes = None
            self.TruckRoutes = []
            self.NrVanRoutes = None
            self.VanRoutes = []

        def __str__(self):
            strRepr = 'Day: %d' % self.dayNumber
            if self.NrTruckRoutes is not None:
                strRepr += '\nNr Truck Routes: %d' % self.NrTruckRoutes
                for i in range(len(self.TruckRoutes)):
                    strRepr += '\n%s' % ( str(self.TruckRoutes[i]) )
            if self.NrVanRoutes is not None:
                strRepr += '\nNr Van Routes: %d' % self.NrVanRoutes
                for i in range(len(self.VanRoutes)):
                    strRepr += '\n%s' % ( str(self.VanRoutes[i]) )
            return strRepr

    class FoodOnAHubOnGivenDay(object):
        def __init__(self, hubID, dayNr, daysFresh):
            self.hubID = hubID
            self.dayNumber = dayNr
            self.daysFresh = daysFresh #array with the nr of days each product type stays fresh after it reaches the hub
            self.Stock = [dict() for i in daysFresh ]   # one dictionary per product, storing {days fresh, amount}

        def moveDayForward_ThrowOutExpiredFood(self):
            self.dayNumber += 1
            newStock = []
            productID = 1
            for amountsDict in self.Stock: #one dictionary per product, storing {days fresh, amount}
            # For example, if a product stays fresh for three days it can be put in the van 0, 1 or 2 days after it comes out of the truck.
            #so store the DaysFresh when it gets to the hub, and throw it out once this counter reaches zero
                newAmountsDict = dict() #init
                for nrDaysRemaining in amountsDict:
                    amount = amountsDict[nrDaysRemaining]
                    #print(f'of Product {productID} there was {amount} at the hub that we may still deliver for another {nrDaysRemaining} days. Now updating..'  )
                    if (nrDaysRemaining > 1):
                        newAmountsDict[nrDaysRemaining-1] = amount #reduce the nr of days remaining by one
                    elif amount > 0 :
                        print(f'Warning: {amount} kg of product {productID} goes in the bin on day {self.dayNumber}')
                newStock.append(newAmountsDict)
                productID += 1
            self.Stock = newStock


        def addTruckDelivery(self, amounts):
            if len(self.daysFresh) != len(amounts):
                sys.exit(f'Wrong nr of products found when processing a truckdelivery hub {self.hubID} with amounts = {amounts} while daysFresh = {self.daysFresh}')
            for i in range(len(amounts)):
                dayThisStaysFresh = self.daysFresh[i]
                value = self.Stock[i].get(dayThisStaysFresh) # .get(key) returns none if not found
                if value == None:
                    self.Stock[i][dayThisStaysFresh] = amounts[i] #store this amount
                else:
                    self.Stock[i][dayThisStaysFresh] = value + amounts[i] #add this amount
                    print(f'adding {amounts[i] } to already existing stock = {value} with dayThisStaysFresh = {dayThisStaysFresh}')


        def subtractVanLoad(self, loading_amounts):
            for i in range(len(loading_amounts)):
                amount_to_process_of_this_product = loading_amounts[i]
                if amount_to_process_of_this_product > 0:
                    oldStock = self.Stock[i]
                    newStock = dict()
                    for nrDaysRemaining in sorted(oldStock): #the sorted thing here ensures FIFO
                        take_from_here = min(amount_to_process_of_this_product, oldStock[nrDaysRemaining])
                        newStock[nrDaysRemaining] = oldStock[nrDaysRemaining] - take_from_here
                        amount_to_process_of_this_product -= take_from_here
                        #if amount_to_process_of_this_product == 0:
                            #print( f'I found ENOUGH fresh product of productID {i+1} on day {self.dayNumber} at hub {self.hubID}')
                            #cannot break because still need to store the other ones in the newStock.
                    self.Stock[i] = newStock
                    if amount_to_process_of_this_product > 0:
                        print(f'Error: There is not enough fresh product of productID {i+1} on day {self.dayNumber} at hub {self.hubID}. Therefore I cannot process one of the van routes.')
                        sys.exit()

        def __str__(self):
            result = f'Stock at day {self.dayNumber} at hub {self.hubID}: \n'
            for s in self.Stock:
                result += str(s)
                result += '\n'
            return result

    def __str__(self):
        strRepr = 'GivenCost: %s\nCalcCost: %s\nDAYS:' % (str(self.givenCost),str(self.calcCost))
        for day in self.Days:
                strRepr += '\n%s\n' % ( str(day) )
        return strRepr

    def __init__(self, inputfile,Instance,filetype=None):
        self.Instance = Instance
        self.Instance.calculateDistances()
        self._doinit(inputfile,filetype)
        if self.isValid():
            self._calculateSolution()


    def _initData(self):
        self.Days = []
        self.givenCost = self.SolutionCost()
        self.calcCost = self.SolutionCost()


    def _readTextCost(self, fd, lastLineAssignment = None):
        if not lastLineAssignment:
            lastLineAssignment = self._isAssignment(fd)

        field = lastLineAssignment[0]
        member = self.LANG.TXT.costfields.get(field)
        while member:
            value = lastLineAssignment[1]
            value = self._checkInt(field,value)
            self.givenCost.__setattr__(member,value) #store the cost parameter that was read in
            lastLineAssignment = self._isAssignment(fd)
            field = lastLineAssignment[0]
            member = self.LANG.TXT.costfields.get(field)

        if lastLineAssignment is None or lastLineAssignment[0] is None or lastLineAssignment[0] == self.LANG.TXT.day:
            return lastLineAssignment
        self._checkError('Unexpected field: %s.' % lastLineAssignment[0], False)
        return lastLineAssignment

    def _readDay(self, fd, lastLineAssignment, hubstocks):
        for hs in hubstocks:
            hs.moveDayForward_ThrowOutExpiredFood()

        self._checkError('Unexpected string: %s.' % lastLineAssignment[1], lastLineAssignment[0] is not None )
        self._checkError('Unexpected field: %s.' % lastLineAssignment[0],lastLineAssignment[0] == self.LANG.TXT.day)
        newDay = self.SolutionDay(self._checkInt(self.LANG.TXT.day,lastLineAssignment[1]))
        self._checkError('Day number should be positive, found %d.' % (newDay.dayNumber), newDay.dayNumber > 0 )
        self._checkError('Day number should be at most %d, found %d.' % (self.Instance.Days,newDay.dayNumber), newDay.dayNumber <= self.Instance.Days )
        lastDay = self.Days[-1].dayNumber if len(self.Days) > 0 else 0
        self._checkError('Incorrect order of days, found day %d after day %d.' % (newDay.dayNumber, lastDay), newDay.dayNumber > lastDay )
        lastLineAssignment = self._isAssignment(fd) #writes error if the line isn't of the form 'string=number', and returns the number
        #read truck routes
        self._checkError('Unexpected field: %s.' % lastLineAssignment[0],lastLineAssignment[0] == self.LANG.TXT.dayNrTrucks)
        newDay.NrTruckRoutes = self._checkInt(self.LANG.TXT.dayNrTrucks,lastLineAssignment[1])
        self._checkError('Nr Trucks used should be non-negative, found %d on day %d.' % (newDay.NrTruckRoutes, newDay.dayNumber), newDay.NrTruckRoutes >= 0 )
        lastLineAssignment = self._isAssignment(fd) #goes to next line and returns a tuple of the thing before and after the = sign. (if not found: returns None, the line)
        nrTruckRoutesFound = 0

        while lastLineAssignment is not None and lastLineAssignment[0] is None: # breaks out of this while loop as soon as an assignment, eg NUMBER_OF_VANS = 3 is found
            nrTruckRoutesFound += 1
            line = lastLineAssignment[1] #if we got here, lastLineAssignment is of the shape (None, line)
            routeLine = line.split()
            truckRoute = self.TruckRoute()
            truckRoute.ID = self._checkInt('Truck ID',routeLine[0]) #the line starts with the truck id

            try:
                for index in range(1,len(routeLine)-1,2) : #use steps of two
                    # we will find e.g. H2 0,2,8 H1 2,4,3
                    hubID = self._checkHub(routeLine[index]) #returns integer
                    truckRoute.Route.append(hubID)
                    amountsString = routeLine[index+1]
                    amountsForThisStop = []
                    for a in amountsString.split(','):
                        a = self._checkInt('A wrong amount was found between commas: ', a)
                        amountsForThisStop.append(a)
                    #old: truckRoute.Route = [int(x) for x in routeLine[2:]]
                    truckRoute.Amounts.append(amountsForThisStop)
                    hubstocks[hubID-1].addTruckDelivery(amountsForThisStop) #keeps track of stock & freshness at hub

            except:
                self._checkError('Error parsing a truckroute on day %d. \n Found incorrect data:\n %s.' % (newDay.dayNumber,line),False)
            self._checkError('Route should be at least length 1, found %d (day %d).' % (len(truckRoute.Route),newDay.dayNumber),len(truckRoute.Route)>=1)
            newDay.TruckRoutes.append(truckRoute)
            lastLineAssignment = self._isAssignment(fd) #gets next line. if it's a truckroute, it returns (None, the line)
        self._checkWarning('Expected %d routes (day %d). Found %d.' % (newDay.NrTruckRoutes, newDay.dayNumber, nrTruckRoutesFound), nrTruckRoutesFound == newDay.NrTruckRoutes)

        #read van routes
        self._checkError('Unexpected field: %s.' % lastLineAssignment[0],lastLineAssignment[0] == self.LANG.TXT.dayNrVans)
        newDay.NrVanRoutes = self._checkInt(self.LANG.TXT.dayNrVans,lastLineAssignment[1])
        self._checkError('Nr of vans used should be non-negative, found %d on day %d.' % (newDay.NrVanRoutes, newDay.dayNumber), newDay.NrVanRoutes >= 0 )
        lastLineAssignment = self._isAssignment(fd)

        nrVanRoutesFound = 0
        while lastLineAssignment is not None and lastLineAssignment[0] is None:
            nrVanRoutesFound += 1
            line = lastLineAssignment[1]
            routeLine = line.split()
            vanRoute = self.VanRoute()
            vanRoute.ID = self._checkInt('Van ID',routeLine[0])
            try:
                vanRoute.HubID = self._checkHub(routeLine[1]) #removes the H and returns the integer
                vanRoute.Route = [int(x) for x in routeLine[2:]]
                amountsInThisVan = [0 for p in self.Instance.Products] #init

                for i in vanRoute.Route:
                    self._checkError('Expected strictly positive integers on the route line (day %d). Found incorrect data: %s.' % (newDay.dayNumber,line),i > 0)
                    req = self.Instance.Requests[i-1]
                    self._checkError('The requests are in the wrong order, I found this on (day %d). Found incorrect data: %s.' % (newDay.dayNumber,line),req.ID == i) #i represents a requestID
                    for product_index in range(len(req.amounts)):
                        amountsInThisVan[product_index] += req.amounts[product_index]
                hubstocks[vanRoute.HubID-1].subtractVanLoad(amountsInThisVan)
            except:
                self._checkError('There is a problem with a van route on day %d. This line is the problem: %s.' % (newDay.dayNumber,line),False)
            self._checkError('Route should be at least length 1, found %d (day %d).' % (len(vanRoute.Route),newDay.dayNumber),len(vanRoute.Route)>=1)
            newDay.VanRoutes.append(vanRoute)
            lastLineAssignment = self._isAssignment(fd)
        self._checkWarning('Expected %d routes (day %d). Found %d.' % (newDay.NrVanRoutes, newDay.dayNumber, nrVanRoutesFound), nrVanRoutesFound == newDay.NrVanRoutes)

        self.Days.append(newDay)
        return lastLineAssignment

    def _initTXT(self):
        try:
            fd = open(self.inputfile, 'r')
        except:
            self.errorReport.append( 'Solution file %s could not be read.' % self.inputfile )
            return

        try:
            with fd:
                self.Dataset = self._checkAssignment(fd,self.LANG.TXT.dataset,'string')

                #initialize the stock at every hub at zero:
                daysFresh = [p.daysFresh for p in self.Instance.Products]
                hubstocks = [self.FoodOnAHubOnGivenDay(h.ID, 0, daysFresh) for h in self.Instance.Hubs]

                assignment = self._readTextCost(fd)
                while assignment:
                    assignment = self._readDay(fd,assignment,hubstocks)


        except self.BaseParseException:
            pass
        except:
            print('Crash during solution reading\nThe following errors were found:')
            print( '\t' + '\n\t'.join(self.errorReport) )
            raise

    def _calculateSolution(self):
        try:
            totalVanDistance = 0
            hubCost = 0
            totalCost = 0
            TruckDistancePerDay = [0] * (len(self.Days))
            NrTrucksPerDay = [0] * (len(self.Days))
            VanDistancePerDay = [0] * (len(self.Days))
            NrVansPerDay = [0] * (len(self.Days))
            self.HubsUsed = set(()) #use a set because it cannot contain duplicates (you pay only once for using a hub at all)

            requestDelivered = [None] * (len(self.Instance.Requests) + 1 )
            for day in self.Days:
                truckDistanceToday = 0
                # Compute truck routes
                NrTrucksPerDay[day.dayNumber - 1] = len(day.TruckRoutes)
                for i in range(len(day.TruckRoutes)):
                    truck = day.TruckRoutes[i]
                    truckDistance = 0 #of only this route
                    truckLoad = 0
                    lastHub = None
                    for stop in range(len(truck.Route)):
                        hub = truck.Route[stop]
                        amounts = truck.Amounts[stop]
                        #if hub == 0: #  depot
                            #truckLoad = 0 #if reloading is allowed at depot
                        if hub > 0:
                            self._checkError('Unknown hub %d (current day %d).' % (hub,day.dayNumber), hub <= len(self.Instance.Hubs) )
                            #self._checkError('Deliver of request %d is already planned on day %d (current day %d).' % (hub, requestDelivered[node] if requestDelivered[node] is not None else 0,day.dayNumber), requestDelivered[node] == None )
                            #add that at day.dayNumber, similar to we did before:
                            #requestDelivered[node] = day.dayNumber
                            truckLoad += sum(amounts)
                            self._checkError('Truckload of truck %d exceeds capacity on day %d' %(truck.ID, day.dayNumber), truckLoad <= self.Instance.TruckCapacity)
                        if lastHub is None:
                            fromCoord = 0 #depot
                            self._checkError('Found a hubs and its ID %d that I dont understand on day %d ' %(hub, day.dayNumber), hub == self.Instance.Hubs[hub-1].ID)
                            toCoord = hub # a hub's location is at the index locations[hubid]
                        elif lastHub is not None:
                            fromCoord = lastHub #the hubID is equal to the index of that hub's location
                            toCoord = hub
                        truckDistance += self.Instance.calcDistance[fromCoord][toCoord]
                        lastHub = hub
                    #From last hub to depot:
                    fromCoord = lastHub
                    self._checkError('found a truckroute for which I dont understand the last hubs (%d) on day %d ' %(hub, day.dayNumber), truck.Route[-1]  == lastHub) #just a sanity check. may remove.
                    toCoord = 0 #always return to depot
                    truckDistance += self.Instance.calcDistance[fromCoord][toCoord]
                    self._checkError('Distance traveled by truck %d  exceeds maximum allowed distance on day %d (%d > %d)' %(day.TruckRoutes[i].ID, day.dayNumber, truckDistance, self.Instance.TruckMaxDistance), truckDistance <= self.Instance.TruckMaxDistance)
                    truckDistanceToday += truckDistance
                TruckDistancePerDay[day.dayNumber - 1] = truckDistanceToday

                # Compute van routes
                NrVansPerDay[day.dayNumber - 1] = len(day.VanRoutes)
                vanDistanceToday = 0
                for VanRoute in day.VanRoutes:
                    self.HubsUsed.add(VanRoute.HubID) #store this so we know to add the costs later

                    #no longer needed: nrOfStops = len(VanRoute.Route)vanDistance = 0
                    vanDistance = 0 #of this route only
                    lastNode = None
                    for node in VanRoute.Route:
                        if node > 0:
                            self._checkError('Unknown request %d (current day %d).' % (node,day.dayNumber), node < len(requestDelivered) ) #cj do we need this?
                            self._checkError('Van delivery of request %d is already planned on day %d (current day %d).' % (node, requestDelivered[node] if requestDelivered[node] is not None else 0,day.dayNumber), requestDelivered[node] == None )
                            self._checkError('Delivery of request %d is not allowed from hub %d.' % (node, VanRoute.HubID), node in self.Instance.Hubs[VanRoute.HubID-1].allowedRequests )
                            requestDelivered[node] = day.dayNumber #keep track of when you deliver. This helps to compute the penalty later
                            if self.Instance.deliverEarlyPenalty == 0: #this should be interpreted as: may NOT deliver early
                                self._checkError('Van delivery of request %d is planned on day %d, but may not take place on any other day than the desired day %d' %(node, day.dayNumber, self.Instance.Requests[node-1].desiredDay), requestDelivered[node] ==  self.Instance.Requests[node-1].desiredDay)
                            else:
                                self._checkError('Van delivery of request %d is planned on day %d but cannot take place after desired day' %(node, day.dayNumber), requestDelivered[node] <=  self.Instance.Requests[node-1].desiredDay)

                        if lastNode is None:
                            fromCoord = VanRoute.HubID #start route at correct hub. hub ID is equal to the index in the coordinates
                            toCoord = self.Instance.Requests[node-1].customerLocID - 1
                        elif lastNode is not None:
                            fromCoord = self.Instance.Requests[lastNode-1].customerLocID - 1
                            toCoord = self.Instance.Requests[node-1].customerLocID - 1
                        vanDistance += self.Instance.calcDistance[fromCoord][toCoord]
                        lastNode = node
                    #Final part of this van trip:
                    fromCoord = self.Instance.Requests[VanRoute.Route[-1] - 1].customerLocID - 1
                    toCoord = VanRoute.HubID #back to the hub
                    vanDistance += self.Instance.calcDistance[fromCoord][toCoord]
                    self._checkError('Distance traveled by van %d exceeds maximum allowed distance on day %d (%d > %d)' %(VanRoute.ID, day.dayNumber, vanDistance, self.Instance.VanMaxDistance), vanDistance <= self.Instance.VanMaxDistance)
                    vanDistanceToday += vanDistance
                VanDistancePerDay[day.dayNumber - 1] = vanDistanceToday

            if self.calcCost.CumulativeEarlyPenalty == None:
                self.calcCost.CumulativeEarlyPenalty = 0

            #Check if all requests are delivered, (we already checked if it was on a correct day beforehand)
            for request in self.Instance.Requests:
                isDelivered = requestDelivered[request.ID] > 0 if requestDelivered[request.ID] is not None else False
                if not isDelivered:
                    self._checkError('Request %d has not been delivered' %request.ID, False)
                else:
                    daysEarly = request.desiredDay - requestDelivered[request.ID]
                    if daysEarly > 0: ##penalty if early delivery is allowed (if not allowed, this has already been dectected earlier)
                        self.calcCost.CumulativeEarlyPenalty += self.Instance.deliverEarlyPenalty ** daysEarly

            #add hubCost for each hub that was opened (these are stored in the set HubsUsed)
            self.calcCost.HubCost = 0
            for used_hubID in self.HubsUsed:
                #find the hub with this id:
                for h in self.Instance.Hubs:
                    if (h.ID == used_hubID):
                        self.calcCost.HubCost += h.hubOpeningCost

            #for day in self.Days:

            totalCost = sum(TruckDistancePerDay) * self.Instance.TruckDistanceCost \
                        + sum(NrTrucksPerDay) * self.Instance.TruckDayCost \
                        + max(NrTrucksPerDay) * self.Instance.TruckCost \
                        + sum(VanDistancePerDay) * self.Instance.VanDistanceCost \
                        + sum(NrVansPerDay) * self.Instance.VanDayCost \
                        + max(NrVansPerDay) * self.Instance.VanCost \
                        + self.calcCost.HubCost \
                        + self.calcCost.CumulativeEarlyPenalty

            self.calcCost.TruckDistance = sum(TruckDistancePerDay)
            self.calcCost.NrTruckDays = sum(NrTrucksPerDay)
            self.calcCost.NrTrucksUsed = max(NrTrucksPerDay)
            self.calcCost.VanDistance = sum(VanDistancePerDay)
            self.calcCost.NrVanDays = sum(NrVansPerDay)
            self.calcCost.NrVansUsed = max(NrVansPerDay)
            #self.calcCost.CumulativeEarlyPenalty is already filled
            self.calcCost.Cost = totalCost

            #this seems convenient info:
            print(f'VanDistancePerDay={VanDistancePerDay}')
            print(f'TruckDistancePerDay={TruckDistancePerDay}')
            print(f'HubsUsed ={self.HubsUsed}')

        except self.BaseParseException:
            pass
        except:
            print('Crash during solution calculation\nThe following errors were found:')
            print( '\t' + '\n\t'.join(self.errorReport) )
            raise


    def isValid(self):
        return not self.errorReport

    def areGivenValuesValid(self):
        result = True
        ''' #turned this off:
        try:
            result = result and self._checkWarning('Incorrect truck distance (given value: %d. Calculated value: %d).' % (self.givenCost.TruckDistance if self.givenCost.TruckDistance is not None else 0, self.calcCost.TruckDistance), self.givenCost.TruckDistance is None or self.givenCost.TruckDistance == self.calcCost.TruckDistance )
            result = result and self._checkWarning('Incorrect number of truck days (given value: %d. Calculated value: %d).' % (self.givenCost.NrTruckDays if self.givenCost.NrTruckDays is not None else 0,self.calcCost.NrTruckDays), self.givenCost.NrTruckDays is None or self.givenCost.NrTruckDays == self.calcCost.NrTruckDays )
            result = result and self._checkWarning('Incorrect number of trucks used (given value: %d. Calculated value: %d).' % (self.givenCost.NrTrucksUsed if self.givenCost.NrTrucksUsed is not None else 0,self.calcCost.NrTrucksUsed), self.givenCost.NrTrucksUsed is None or self.givenCost.NrTrucksUsed == self.calcCost.NrTrucksUsed )
            result = result and self._checkWarning('Incorrect van distance (given value: %d. Calculated value: %d).' % (self.givenCost.VanDistance if self.givenCost.VanDistance is not None else 0,self.calcCost.VanDistance), self.givenCost.VanDistance is None or self.givenCost.VanDistance == self.calcCost.VanDistance )
            result = result and self._checkWarning('Incorrect number of van days (given value: %d. Calculated value: %d).' % (self.givenCost.NrVanDays if self.givenCost.NrVanDays is not None else 0,self.calcCost.NrVanDays), self.givenCost.NrVanDays is None or self.givenCost.NrVanDays == self.calcCost.NrVanDays )
            result = result and self._checkWarning('Incorrect number of vans used (given value: %d. Calculated value: %d).' % (self.givenCost.NrVansUsed if self.givenCost.NrVansUsed is not None else 0,self.calcCost.NrVansUsed), self.givenCost.NrVansUsed is None or self.givenCost.NrVansUsed == self.calcCost.NrVansUsed )
            result = result and self._checkWarning('Incorrect hub cost (given value: %d. Calculated value: %d).' % (self.givenCost.HubCost if self.givenCost.HubCost is not None else 0,self.calcCost.HubCost), self.givenCost.HubCost is None or self.givenCost.HubCost == self.calcCost.HubCost )
            result = result and self._checkWarning('Incorrect total cost (given value: %d. Calculated value: %d).' % (self.givenCost.Cost if self.givenCost.Cost is not None else 0,self.calcCost.Cost), self.givenCost.Cost is None or self.givenCost.Cost == self.calcCost.Cost )

        except self.BaseParseException as E:
            return (False, E.message if E.message is not None else '')
        except:
            print('Crash during solution validation\nThe following errors were found:')
            print( '\t' + '\n\t'.join(self.errorReport) )
            raise
        '''
        return (result, '')


def DoWork(args):
    instance = args.instance
    if not instance:
        print('No instance file specified.')
        return

    Instance = InstanceCO22(instance,args.itype)
    if not Instance.isValid():
        print('File %s is an invalid instance file\nIt contains the following errors:' % instance)
        print( '\t' + '\n\t'.join(Instance.errorReport) )
        return
    Solution = SolutionCO22(args.solution,Instance,args.type)
    if Solution.isValid():
        print('Solution %s is a valid solution' % args.solution)
        print('\t' + '\n\t'.join(str(Solution.calcCost).split('\n')))

        if len(Solution.warningReport) > 0:
            print('There were warnings:')
            print( '\t' + '\n\t'.join(Solution.warningReport) )
    else:
        print('File %s is an invalid solution file\nIt contains the following errors:' % args.solution)
        print( '\t' + '\n\t'.join(Solution.errorReport) )
        if len(Solution.warningReport) > 0:
            print('There were also warnings:')
            print( '\t' + '\n\t'.join(Solution.warningReport) )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read and checks solution file.')
    parser.add_argument('--solution', '-s', metavar='SOLUTION_FILE', required=True,help='The solution file')
    parser.add_argument('--instance', '-i', metavar='INSTANCE_FILE', help='The instance file')
    parser.add_argument('--type', '-t', choices=['txt', 'xml'],                    help='Solution file type')
    parser.add_argument('--itype', choices=['txt', 'xml'],                         help='instance file type')
    parser.add_argument('--outputFile', '-o', metavar='NEW_SOLUTION_FILE',
                        help='Write the solution to this file')  #SimpleSolver can use this
    #parser.add_argument('--writeExtra', '-e', action='store_true',                         help='Write the extra data in the outputfile')
    #parser.add_argument('--skipExtraDataCheck', '-S', action='store_true',                 help='Skip extra data check')
    #parser.add_argument('--continueOnError', '-C', action='store_true',                    help='Try to continue after the first error in the solution. This may result in a crash (found errors are reported). Note: Any error after the first may be a result of a previous error')
    args = parser.parse_args()

    DoWork(args)
