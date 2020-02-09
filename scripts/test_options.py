# ***********************************************************************                                                                   
# Import libraries                                                                                                                          
# ***********************************************************************

import sys
import os
import dill
import datetime
import time
import numpy as np
import pandas as pd

sys.path.append( os.path.abspath( '../' ) )

import utl.utils as utl

from prt.prt import MfdOptionsPrt

# ***********************************************************************                                                                   
# Input parameters
# ***********************************************************************

modFile = 'models/model_long_term.dill'
curDate = '2020-02-08'
maxDate = '2020-09-30'

indices = [ 'INDU', 'NDX', 'SPX', 'RUT', 'OEX',
            'MID',  'SOX', 'RUI', 'RUA', 'TRAN',
            'HGX',  'TYX', 'XAU' ]

futures = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

ETFs    = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH',
            'SMH', 'XLE', 'XLF', 'XLU', 'EWJ' ]

# ***********************************************************************                                                                   
# Get asset prices
# ***********************************************************************

# print( 'Getting assetHash...' )

# t0 = time.time()

# assetHash = {}

# for symbol in ETFs:
#     val, date = utl.getKibotLastValue( symbol,
#                                        sType = 'ETF' )
#     assetHash[ symbol ] = val

# for symbol in futures:
#     val, date = utl.getKibotLastValue( symbol,
#                                        sType = 'futures' )
#     assetHash[ symbol ] = val

# for symbol in indices:
#     val, date = utl.getKibotLastValue( symbol,
#                                        sType = 'index' )
#     assetHash[ symbol ] = val

# print( 'Done with getting assetHash! Took %0.2f seconds!' % ( time.time() - t0 ) )

# print( assetHash )

assetHash = {'QQQ': 228.83, 'SPY': 331.72, 'DIA': 290.99, 'MDY': 373.47, 'IWM': 164.76, 'OIH': 11.14, 'SMH': 143.05, 'XLE': 53.96, 'XLF': 30.86, 'XLU': 68.59, 'EWJ': 59.5, 'ES': 3322.75, 'NQ': 9401.0, 'US': 162.46875, 'YM': 29031, 'RTY': 1656.8, 'EMD': 2048.7, 'QM': 50.35, 'INDU': 29102.51, 'NDX': 9401.1, 'SPX': 3327.71, 'RUT': 1656.7779999999998, 'OEX': 1498.07, 'MID': 2049.3, 'SOX': 1864.36, 'RUI': 1839.675, 'RUA': 1946.354, 'TRAN': 10857.73, 'HGX': 372.58, 'TYX': 20.42, 'XAU': 101.33}

# ***********************************************************************                                                                   
# Get options chain
# ***********************************************************************

options = []

for symbol in ETFs:
    
    print( 'Getting options for %s...' % symbol )
    
    tmpList = utl.getOptionsChain( symbol,
                                   minExprDate  = pd.to_datetime( curDate ) + datetime.timedelta( days = 7 ),
                                   maxExprDate  = maxDate,
                                   minTradeDate = pd.to_datetime( curDate ) - datetime.timedelta( days = 2 ),
                                   minVolume    = 0,
                                   minInterest  = 0,
                                   maxTries     = 2,
                                   logger       = None   )
    options += tmpList
    
print( 'Found %d options contracts!' % len( options ) )

# ***********************************************************************                                                                   
# Process options
# ***********************************************************************

prtObj = MfdOptionsPrt( modFile     = modFile,
                        assetHash   = assetHash,
                        curDate     = curDate,
                        maxDate     = maxDate,
                        rfiDaily    = 0.0,
                        tradeFee    = 0.0,
                        nDayTimes   = 1140,
                        minProb     = 0.5,
                        logFileName = None,                    
                        verbose     = 1          )                        

sOptions, probs = prtObj.sortOptions( options )

print( sOptions[:5] )
print( probs[:5] )
