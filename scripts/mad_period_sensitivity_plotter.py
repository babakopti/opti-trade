# ***********************************************************************
# Import libraries
# ***********************************************************************

import os
import sys
import dill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    
sys.path.append( os.path.abspath( '../' ) )

from prt.prt import MfdPrt

# ***********************************************************************
# Set some parameters
# ***********************************************************************

madPeriods  = [ 360, 180, 90, 60, 45, 30, 21, 14, 7 ]
madOptTols  = [ 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2] 

study       = 'SPY'
fallBack    = 'macd'
modFilePath = 'models/model_2018-05-10 00:00:00.dill'
figName     = 'figures/mad-sensitivity-period-' + study + '.png'

ETFs        = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH', 
                'SMH', 'XLE', 'XLF', 'XLU', 'EWJ'          ]
assets      = ETFs
totAssetVal = 1000000.0
tradeFee    = 6.95
nPrdDays    = 1
nPrdTimes   = int( nPrdDays * 19 * 60 )

# ***********************************************************************
# plot
# ***********************************************************************

xVals = []
yVals = []

for item in madPeriods:

    print( 'Calculating portfolio for MAD period', item, '...' )
    
    mfdPrt = MfdPrt( modFile      = modFilePath,
                     assets       = assets,
                     nRetTimes    = item,
                     nPrdTimes    = nPrdTimes,
                     totAssetVal  = totAssetVal, 
                     tradeFee     = tradeFee,
                     strategy     = 'mad',
                     minProbLong  = 0.5,
                     minProbShort = 0.5,
                     vType        = 'vel',
                     fallBack     = fallBack,
                     optTol       = 1.0e-6,
                     verbose      = 1          )
    
    wtHash = mfdPrt.getPortfolio()

    xVals.append( item )
    yVals.append( wtHash[ study ] )

xVals = np.array( xVals, dtype = 'd' )
yVals = np.array( yVals, dtype = 'd' )

sortDict = {}
for j in range( len( xVals ) ):
    sortDict[ yVals[j] ] = xVals[j] 

xVals = sorted( xVals )
yVals = sorted( yVals, key = lambda y : sortDict[y] )
    
plt.plot( xVals, yVals, 'o-' )
plt.xlabel( 'MAD period (days)' )
plt.ylabel( 'wt. of ' + study )
plt.savefig( figName )
plt.show()
