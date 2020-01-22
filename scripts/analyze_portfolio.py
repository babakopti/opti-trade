# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import datetime
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append( '../' )

import utl.utils as utl

# ***********************************************************************
# Input
# ***********************************************************************

outFile     = 'analysis-results/backtest_returns_sort_ETFs_60_eval_days.csv'
prtFile     = 'portfolios/sorted_ETFs_portfolio_60_eval_days.txt'
dfFile      = 'data/dfFile_kibot_2016plus.pkl'
base        = 'SPY'
initTotVal  = 1000000.0

invHash = {   'TQQQ' : 'SQQQ',
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
# Read portfolio dates, assets
# ***********************************************************************

prtWtsHash = ast.literal_eval( open( prtFile, 'r' ).read() )

# ***********************************************************************
# Get actual open prices
# ***********************************************************************

retDf = utl.calcBacktestReturns(  prtWtsHash = prtWtsHash,
                                  dfFile     = dfFile,
                                  initTotVal = initTotVal,
                                  shortFlag  = False,
                                  invHash    = invHash   )

print( retDf.head(10) )

retDf[ 'isPos' ] = retDf.Return.apply( lambda x : 1 if x > 0 else -1 )

retDf.to_csv( outFile, index = False )
    
tmpDf1 = retDf[ retDf.Return > 0 ]
tmpDf2 = retDf[ retDf.Return <= 0 ]

plt.plot( tmpDf1.Match, tmpDf1.Return, 'go',
          tmpDf2.Match, tmpDf2.Return, 'ro' )

plt.xlabel( 'Trend Match Ratio' )
plt.ylabel( 'Daily Return (%)' )
plt.show()
