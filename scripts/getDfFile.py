# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import datetime
import numpy as np
import pandas as pd

sys.path.append( os.path.abspath( '../' ) )

import utl.utils as utl

from dat.assets import ETF_HASH, SUB_ETF_HASH, FUTURES

# ***********************************************************************
# Set some parameters and read data
# ***********************************************************************

etfs    = list( ETF_HASH.keys() ) + list( ETF_HASH.values() )
stocks  = []
futures = FUTURES
indexes = []

minDate = pd.to_datetime( '2018-12-01 00:00:00' )

symbols    = etfs + stocks + futures + indexes
baseDatDir = '/var/data'
dfFile     = 'data/dfFile_2020.pkl'

# ***********************************************************************
# Get data and save to pickle file
# ***********************************************************************

oldDf = utl.mergeSymbols( symbols = symbols,
                          datDir  = baseDatDir,
                          fileExt = 'pkl',
                          minDate = minDate,
                          logger  = None   )

newDf = utl.getYahooData( etfs    = etfs,
                          stocks  = stocks,
                          futures = futures,
                          indexes = indexes,
                          nDays   = 5,
                          logger  = None  )
        
newDf = newDf[ newDf.Date > oldDf.Date.max() ]
newDf = pd.concat( [ oldDf, newDf ] )

newDf.to_pickle( dfFile, protocol = 4 )
