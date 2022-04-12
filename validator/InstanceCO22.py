#! /usr/bin/env python
"""

@author: caroline jagtenberg based on ortec verologsolver
"""

import argparse
import math
try:
    import baseParser as base
except:
    import validator.baseParser as base


class InstanceCO22(base.BaseParser):
    parsertype = 'instance'

    class LANG:
        class TXT:
            dataset = 'DATASET'
            #distance = 'DISTANCE' not sure if we need this
            days = 'DAYS'
            truckCapacity = 'TRUCK_CAPACITY'
            truckMaxDistance = 'TRUCK_MAX_DISTANCE'
            vanCapacity = 'VAN_CAPACITY'
            vanMaxDistance = 'VAN_MAX_DISTANCE'

            truckDistanceCost = 'TRUCK_DISTANCE_COST'
            truckDayCost = 'TRUCK_DAY_COST'
            truckCost = 'TRUCK_COST'

            vanDistanceCost = 'VAN_DISTANCE_COST'
            vanDayCost = 'VAN_DAY_COST'
            vanCost = 'VAN_COST'
            deliverEarlyPenalty = 'DELIVER_EARLY_PENALTY'
            products = 'PRODUCTS'
            hubs= 'HUBS'
            locations = 'LOCATIONS'
            requests = 'REQUESTS'

    class Hub(object):
        def __init__(self,ID,hubOpeningCost,allowedRequests):
            self.ID = ID
            self.hubOpeningCost = hubOpeningCost
            self.allowedRequests = []
            for a in allowedRequests.split(','):
                self.allowedRequests.append(int(a))  #translates e.g. the string 1,2,4,5 to an array of ints

        def __repr__(self):
            result = f'hub {self.ID} with cost {self.hubOpeningCost} and allowed requests '
            for a in self.allowedRequests:
                result += str(a)
            return result             #'{:>5} {:>5}'.format(self.ID,self.allowedRequests)

    class Product(object):
        def __init__(self,ID,daysFresh):
            self.ID = ID
            self.daysFresh = daysFresh

        def __repr__(self):
            return '{:>5} {:>5}'.format(self.ID,self.daysFresh)

    class Request(object):
        def __init__(self,ID,desiredDay,customerLocID,kgArray):
            self.ID = ID
            self.desiredDay = desiredDay
            self.customerLocID = customerLocID
            self.amounts = []
            for kg in kgArray.split(','):
                self.amounts.append(int(kg))

        def __repr__(self):
            result = 'Id:{:>2} desiredDay{:>2} customerLocID{:>2}'.format(self.ID,self.desiredDay,self.customerLocID)
            result += f' amounts = '
            for i in self.amounts:
                result += str(i)
                result += ','
            return result + "\n"

    class Location(object):
        def __init__(self,ID,X,Y):
            self.ID = ID
            self.X = X
            self.Y = Y

        def __repr__(self):
            return '{:>5} {:>5} {:>5}'.format(self.ID,self.X,self.Y)

    def __init__(self, inputfile=None,filetype=None,continueOnErr=False):
        if inputfile is not None:
            self._doinit(inputfile,filetype,continueOnErr)
        else:
            self._initData()

    def _initData(self):
        self.Products = []
        self.Requests = []
        self.Locations = []
        self.Hubs = []
        #self.ReadDistance = None
        self.calcDistance = None

    def _initTXT(self):
        try:
            fd = open(self.inputfile, 'r')
        except:
            self.errorReport.append( 'Instance file %s could not be read.' % self.inputfile )
            return
        try:
            with fd: #the elements are expected exactly in this order:
                self.Dataset = self._checkAssignment(fd,self.LANG.TXT.dataset,'string')
                self.Days = self._checkInt( 'Days', self._checkAssignment(fd,self.LANG.TXT.days) )
                self.TruckCapacity = self._checkInt( 'Truck Capacity', self._checkAssignment(fd,self.LANG.TXT.truckCapacity) )
                self.TruckMaxDistance = self._checkInt( 'Truck Max trip distance', self._checkAssignment(fd,self.LANG.TXT.truckMaxDistance) )
                self.VanCapacity = self._checkInt( 'Truck Capacity', self._checkAssignment(fd,self.LANG.TXT.vanCapacity) )
                self.VanMaxDistance = self._checkInt( 'Truck Max trip distance', self._checkAssignment(fd,self.LANG.TXT.vanMaxDistance) )
                self.TruckDistanceCost = self._checkInt( 'Truck Distance Cost', self._checkAssignment(fd,self.LANG.TXT.truckDistanceCost) )
                self.TruckDayCost = self._checkInt( 'Truck Day Cost', self._checkAssignment(fd,self.LANG.TXT.truckDayCost) )
                self.TruckCost = self._checkInt( 'Truck Cost', self._checkAssignment(fd,self.LANG.TXT.truckCost) )
                self.VanDistanceCost = self._checkInt( 'Van Distance Cost', self._checkAssignment(fd,self.LANG.TXT.vanDistanceCost) )
                self.VanDayCost = self._checkInt( 'Van Day Cost', self._checkAssignment(fd,self.LANG.TXT.vanDayCost) )
                self.VanCost = self._checkInt( 'Van Cost', self._checkAssignment(fd,self.LANG.TXT.vanCost) )
                self.deliverEarlyPenalty = self._checkInt( 'Deliver Early Penalty', self._checkAssignment(fd,self.LANG.TXT.deliverEarlyPenalty) )

                nrProducts = self._checkInt("Number of products", self._checkAssignment(fd,self.LANG.TXT.products))
                for i in range(nrProducts):
                    line = self._getNextLine(fd)
                    ProductLine = line.split()
                    self._checkError("Expected 2 integers on a product line. Found: '%s'." % line,  len(ProductLine) == 2)
                    productID = self._checkInt('Product ID', ProductLine[0] )
                    daysFresh = self._checkInt('daysFresh', ProductLine[1], 'for product %d ' % productID )
                    self.Products.append( self.Product(productID,daysFresh) )
                    self._checkError('The indexing of the Products is incorrect at Product nr. %d.' % productID, productID == len(self.Products) )

                nrHubs = self._checkInt("Number of hubs", self._checkAssignment(fd,self.LANG.TXT.hubs))
                for i in range(nrHubs):
                    line = self._getNextLine(fd)
                    HubLine = line.split() #divides a string into a list. The separator is a whitespace by default
                    hubID = self._checkInt('Hub ID', HubLine[0] )
                    hubOpeningCost = self._checkInt('Hub opening cost', HubLine[1] )
                    allowedRequests = HubLine[2]
                    for a in allowedRequests.split(','):
                        a = self._checkInt('In a Hub line a wrong amount is mentioned', a)
                    self.Hubs.append( self.Hub(hubID,hubOpeningCost,allowedRequests) )

                nrLocations = self._checkInt("Number of locations", self._checkAssignment(fd,self.LANG.TXT.locations))
                for i in range(nrLocations):
                    line = self._getNextLine(fd)
                    LocationLine = line.split()
                    self._checkError("Expected three integers on a coordinate line. Found: '%s'." % line,
                                    len(LocationLine) == 3)
                    locID = self._checkInt('Coordinate ID', LocationLine[0] )
                    X = self._checkInt('Coordinate X', LocationLine[1], 'for Location %d ' % locID )
                    Y = self._checkInt('Coordinate Y', LocationLine[2], 'for Location %d ' % locID )
                    self.Locations.append( self.Location(locID,X,Y) )
                    self._checkError('The indexing of the Locations is incorrect at Location nr. %d.' % locID, locID == len(self.Locations) )

                nrRequests = self._checkInt("Number of requests", self._checkAssignment(fd,self.LANG.TXT.requests))
                for i in range(nrRequests):
                    line = self._getNextLine(fd)
                    RequestLine = line.split()
                    self._checkError("Expected four entries (separated by spaces) on a request line. Found: '%s'." % line,
                                    len(RequestLine) == 4)
                    requestID = self._checkInt('Request ID', RequestLine[0] )
                    desiredDay = self._checkInt('Request from-day', RequestLine[1], 'for Request %d ' % requestID )
                    self._checkError('Request from-day %d is larger than the horizon (%d) for request %d' % (desiredDay, self.Days, requestID), 0 < desiredDay <= self.Days )
                    customerLocID = self._checkInt('Customer Location ID', RequestLine[2], 'for Request %d ' % requestID )
                    self._checkError('Customer Location ID %d for request %d is larger than the number of locations (%d)' % (customerLocID, requestID, nrLocations), 0 < customerLocID <= nrLocations )

                    amounts = RequestLine[3]
                    for a in amounts.split(','):
                        self._checkInt('Requested amounts', a, 'for Request %d ' % requestID )
                        a = int(a) #this conversion is needed otherwise the line below gives an error when comparing a to zero
                        self._checkError('Requested amount is not strictly positive (%d) for request %d' % (a, requestID), 0 <= a ) #need to split the amounts first
                    self.Requests.append( self.Request(requestID,desiredDay,customerLocID,amounts) )
                    self._checkError('The indexing of the Requests is incorrect at Request nr. %d.' % requestID, requestID == len(self.Requests) )
        except:
            print('Crash during Verolog2019 instance reading\nThe following errors were found:')
            print( '\t' + '\n\t'.join(self.errorReport) )
            raise

    def calculateDistances(self): #was used in WriteMatrix which is removed. But will still be useful later
        if not self.isValid() or self.calcDistance is not None:
            return
        numLocs = len(self.Locations)
        self.calcDistance = [[0 for x in range(numLocs)] for x in range(numLocs)]
        for i in range(numLocs):
            cI = self.Locations[i]
            for j in range(i,numLocs):
                cJ = self.Locations[j]
                dist = math.ceil( math.sqrt( pow(cI.X-cJ.X,2) + pow(cI.Y-cJ.Y,2) ) )
                self.calcDistance[i][j] = self.calcDistance[j][i] = int(dist)

    def isValid(self):
        return not hasattr(self, 'errorReport') or not self.errorReport



    def writeInstance(self,filename):
        res = self._writeInstanceTXT(filename)
        if res[0]:
            print('Instance file written to %s' % filename)
        else:
            print('Error writing output file %s: %s' % (filename,res[1]))

    def _writeInstanceTXT(self,filename):
        try:
            fd = open(filename,  mode='w')
        except:
            return (False, 'Could not write to file.')

        with fd:
            self._writeAssignment(fd,self.LANG.TXT.dataset,self.Dataset)
            fd.write('\n')
            self._writeAssignment(fd,self.LANG.TXT.days,self.Days)
            self._writeAssignment(fd,self.LANG.TXT.truckCapacity,self.TruckCapacity)
            self._writeAssignment(fd,self.LANG.TXT.truckMaxDistance,self.TruckMaxDistance)
            fd.write('\n')
            self._writeAssignment(fd,self.LANG.TXT.truckDistanceCost,self.TruckDistanceCost)
            self._writeAssignment(fd,self.LANG.TXT.truckDayCost,self.TruckDayCost)
            self._writeAssignment(fd,self.LANG.TXT.truckCost,self.TruckCost)
            self._writeAssignment(fd,self.LANG.TXT.vanDistanceCost,self.VanDistanceCost)
            self._writeAssignment(fd,self.LANG.TXT.vanDayCost,self.VanDayCost)
            self._writeAssignment(fd,self.LANG.TXT.vanCost,self.VanCost)
            fd.write('\n')

            self._writeAssignment(fd,self.LANG.TXT.products,len(self.Products))
            for i in range(len(self.Products)):
                fd.write('%s\n' % str(self.Products[i]) )
            fd.write('\n')

            self._writeAssignment(fd,self.LANG.TXT.locations,len(self.Locations))
            for i in range(len(self.Locations)):
                fd.write('%s\n' % str(self.Locations[i]) )
            fd.write('\n')

            self._writeAssignment(fd,self.LANG.TXT.requests,len(self.Requests))
            for i in range(len(self.Requests)):
                fd.write('%s\n' % str(self.Requests[i]) )
            fd.write('\n')

            self._writeAssignment(fd,self.LANG.TXT.hubs,len(self.Hubs))
            for i in range(len(self.Hubs)):
                fd.write('%s\n' % str(self.Hubs[i]) )
            fd.write('\n')

        return (True, '')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read and checks Verolog2019 instance file.')
    parser.add_argument('--instance', '-i', metavar='INSTANCE_FILE', required=True,
                        help='The instance file')
    parser.add_argument('--type', '-t', choices=['txt'],
                        help='Instance file type')
    parser.add_argument('--skipDistanceCheck', '-S', action='store_true',
                        help='Skip check on given distances')
    parser.add_argument('--outputFile', '-o', metavar='NEW_INSTANCE_FILE',
                        help='Write the instance to this file')
    #parser.add_argument('--continueOnError', '-C', action='store_true', help='Try to continue after the first error in the solution. This may result in a crash (found errors are reported). Note: Any error after the first may be a result of a previous error')
    args = parser.parse_args()


    Instance = InstanceCO22(args.instance,args.type)
    if Instance.isValid():
        print('Instance %s is a valid CO2022 instance' % args.instance)
        if args.outputFile:
            Instance.writeInstance(args.outputFile)
        if len(Instance.warningReport) > 0:
            print('There were warnings:')
            print( '\t' + '\n\t'.join(Instance.warningReport) )
    else:
        print('File %s is an invalid Verolog2019 instance file\nIt contains the following errors:' % args.instance)
        print( '\t' + '\n\t'.join(Instance.errorReport) )
        if len(Instance.warningReport) > 0:
            print('There were also warnings:')
            print( '\t' + '\n\t'.join(Instance.warningReport) )
