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
dfFile      = 'data/dfFile_kibot_2017plus.pkl'
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

retDf = utl.calcPrtReturns( prtWtsHash = prtWtsHash,
                            dfFile     = dfFile,
                            initTotVal = initTotVal,
                            shortFlag  = False,
                            invHash    = invHash,
                            minDate    = None,
                            maxDate    = None      )

# ***********************************************************************
# Plot
# ***********************************************************************

plt.plot( retDf.Date, retDf.Value )
plt.xlabel( 'Date' )
plt.ylabel( 'Value' )
plt.title( prtFile )
plt.show()
