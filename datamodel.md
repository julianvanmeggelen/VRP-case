## instance solution datamodel
```
{'depotRoutes': 
    {day:                                                   for every day in instance days
        {routeID:                                           for every routeID in [1, ... , #routes] for day
            [{                                              for every node on the route
                'X',
                'Y',
                'locID',
                'collect',
                'demand',
                'amounts',
                'frequency',
                'lower',
                'service_time',
                'upper'
            }]
        }
    },
 'hubRoutes': 
    {day:                                                   for every day in instance days
        {hubLocID:                                          for every hub active on the day  
            {
            'amounts',
            'demand',
            'routes':                                       
                {routeID:                                   for every routeID in [1, ... , #routes] for hub
                    {'route': 
                        [{                                  for every node on the route
                            'X',
                            'Y',
                            'collect',
                            'demand',
                            'frequency',
                            'locID',
                            'reqID',
                            'lower',
                            'service_time',
                            'upper'
                        }]
                    }
                }
            }
        }
    }
}
```                                      