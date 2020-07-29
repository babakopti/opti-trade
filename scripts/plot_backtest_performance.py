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

# ***********************************************************************
# ETF_HASH
# ***********************************************************************

ETF_HASH = {  'TQQQ' : 'SQQQ',
              'SPY'  : 'SH',
              'DDM'  : 'DXD',
              'MVV'  : 'MZZ',
              'UWM'  : 'TWM',
              'SAA'  : 'SDD',
              'UYM'  : 'SMN',
              'UGE'  : 'SZK',
              'UCC'  : 'SCC',
              'FINU' : 'FINZ',
              'RXL'  : 'RXD',
              'UXI'  : 'SIJ',
              'URE'  : 'SRS',
              'ROM'  : 'REW',
              'UJB'  : 'SJB',
              'AGQ'  : 'ZSL',     
              'DIG'  : 'DUG',
              'USD'  : 'SSG',
              'ERX'  : 'ERY',
              'UYG'  : 'SKF',
              'UCO'  : 'SCO',
              'BOIL' : 'KOLD',
              'UPW'  : 'SDP',
              'UGL'  : 'GLL',
              'BIB'  : 'BIS',
              'UST'  : 'PST',
              'UBT'  : 'TBT' }

# ***********************************************************************
# Input
# ***********************************************************************

prtFile     = 'portfolios/FIXED_sorted_ETFs_portfolio_60_eval_days.txt'
prtFile     = 'portfolios/subset_daily_mad_mean_sorted_ETFs_portfolio_60_eval_days.txt'
prtFile     = 'portfolios/sorted_ETFs_portfolio_60_eval_days.txt'
dfFile      = 'data/dfFile_kibot_2016plus.pkl'
base        = 'SPY'
initTotVal  = 20000.0
minDate     = '2017-01-01'

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
                                  hourOset   = 0.,
                                  invHash    = ETF_HASH   )
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

print( retDf1 )

y1 = 100.0 * ( retDf1.EndVal - \
               list( retDf1.EndVal )[0] ) \
        / list( retDf1.EndVal )[0]
y3 = 100.0 * ( retDf3.EndVal - \
               list( retDf3.EndVal )[0] ) \
        / list( retDf3.EndVal )[0]
plt.plot( retDf1.Date, y1, 'b', linewidth = 2 )
plt.plot( retDf3.Date, y3, 'r', linewidth = 2 )
plt.xticks( fontsize = 16 )
plt.yticks( fontsize = 16 )
plt.xlabel( 'Date', fontsize = 18 )
plt.ylabel( 'Return (%)', fontsize = 18 )
plt.legend( [ 'Geometric Algorithm',
              base ],
            fontsize = 18 )
plt.show()
