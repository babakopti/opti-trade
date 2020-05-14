# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import datetime
import ast
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append( '../' )

import utl.utils as utl

from dat.assets import SUB_ETF_HASH as ETF_HASH

# ***********************************************************************
# Input
# ***********************************************************************

prtFile     = 'portfolio_once_a_day_2020.json'
dfFile      = 'data/dfFile_2020.pkl'
base        = 'SPY'
initTotVal  = 20000.0
minDate     = '2020-01-22'

# ***********************************************************************
# Read portfolio dates, assets
# ***********************************************************************

if prtFile.split( '.' )[-1] == 'json':
    prtWtsHash = json.load( open( prtFile, 'r' ) )
else:
    prtWtsHash = ast.literal_eval( open( prtFile, 'r' ).read() )

# ***********************************************************************
# Get actual open prices
# ***********************************************************************

retDf1 = utl.calcBacktestReturns( prtWtsHash = prtWtsHash,
                                  dfFile     = dfFile,
                                  initTotVal = initTotVal,
                                  shortFlag  = False,
                                  minDate    = minDate,
                                  hourOset   = 0.5,
                                  invHash    = ETF_HASH   )

# retDf2 = utl.calcBacktestReturns( prtWtsHash = prtWtsHash,
#                                   dfFile     = dfFile,
#                                   initTotVal = initTotVal,
#                                   shortFlag  = True      )

baseHash = {}

for date in prtWtsHash:
    baseHash[ date ] = { base : 1.0 }

retDf3 = utl.calcBacktestReturns( prtWtsHash = baseHash,
                                  dfFile     = dfFile,
                                  initTotVal = initTotVal,
                                  minDate    = minDate,                                  
                                  shortFlag  = True       )

# ***********************************************************************
# Plot
# ***********************************************************************

plt.plot( retDf1.Date, retDf1.EndVal, 'b',
#          retDf2.Date, retDf2.EndVal, 'g',
          retDf3.Date, retDf3.EndVal, 'r'  )
plt.xlabel( 'Date' )
plt.ylabel( 'Value ($)' )
plt.legend( [ 'Inverse ETF',
#              'Short Sell',
              base ] )
plt.title( prtFile )
plt.show()
