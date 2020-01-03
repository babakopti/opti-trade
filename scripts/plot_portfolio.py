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

prtFile     = 'p_5sortETF_kibot.txt'
dfFile      = 'data/dfFile_kibot_2016plus.pkl'
base        = 'SPY'
initTotVal  = 1000000.0
minDate     = None
maxDate     = None

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

retDf1 = utl.calcPrtReturns( prtWtsHash = prtWtsHash,
                             dfFile     = dfFile,
                             initTotVal = initTotVal,
                             shortFlag  = False,
                             invHash    = invHash,
                             minDate    = minDate,
                             maxDate    = maxDate      )

retDf2 = utl.calcPrtReturns( prtWtsHash = prtWtsHash,
                             dfFile     = dfFile,
                             initTotVal = initTotVal,
                             shortFlag  = True,
                             minDate    = minDate,
                             maxDate    = maxDate      )

baseHash = {}

for date in prtWtsHash:
    baseHash[ date ] = { base : 1.0 }

retDf3 = utl.calcPrtReturns( prtWtsHash = baseHash,
                             dfFile     = dfFile,
                             initTotVal = initTotVal,
                             shortFlag  = True,
                             minDate    = minDate,
                             maxDate    = maxDate      )

# ***********************************************************************
# Plot
# ***********************************************************************

plt.plot( retDf1.Date, retDf1.Value, 'b',
          retDf2.Date, retDf2.Value, 'g',
          retDf3.Date, retDf3.Value, 'r'  )
plt.xlabel( 'Date' )
plt.ylabel( 'Value ($)' )
plt.legend( [ 'Inverse ETFs', 'Short Sell', base ] )
plt.title( prtFile )
plt.show()
