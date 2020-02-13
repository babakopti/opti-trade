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

modFile = 'models/model_long_term_snap_2020_01_31.dill'
curDate = '2020-02-11'
maxDate = '2020-05-30'

indices = [ 'INDU', 'NDX', 'SPX', 'RUT', 'OEX',
            'MID',  'SOX', 'RUI', 'RUA', 'TRAN',
            'HGX',  'TYX', 'XAU' ]

futures = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

ETFs    = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH',
            'SMH', 'XLE', 'XLF', 'XLU', 'EWJ' ]

cash    = 20000

# ***********************************************************************                                                                   
# Get asset prices
# ***********************************************************************

if False:
    assetHash = {'QQQ': 234.35, 'SPY': 337.4, 'DIA': 295.87, 'MDY': 380.93, 'IWM': 168.12, 'OIH': 11.4, 'SMH': 151.0, 'XLE': 55.08, 'XLF': 31.14, 'XLU': 69.2, 'EWJ': 59.443000000000005, 'ES': 3371.75, 'NQ': 9594.25, 'US': 162.0, 'YM': 29445, 'RTY': 1685.7, 'EMD': 2086.4, 'QM': 51.5, 'INDU': 29276.34, 'NDX': 9517.86, 'SPX': 3357.75, 'RUT': 1677.515, 'OEX': 1508.33, 'MID': 2076.67, 'SOX': 1931.08, 'RUI': 1857.118, 'RUA': 1965.181, 'TRAN': 10902.72, 'HGX': 375.83, 'TYX': 20.51, 'XAU': 102.87}

else:
    print( 'Getting assetHash...' )

    t0 = time.time()

    assetHash = {}

    for symbol in ETFs:
        val, date = utl.getKibotLastValue( symbol,
                                           sType = 'ETF' )
        assetHash[ symbol ] = val

    for symbol in futures:
        val, date = utl.getKibotLastValue( symbol,
                                           sType = 'futures' )
        assetHash[ symbol ] = val

    for symbol in indices:
        val, date = utl.getKibotLastValue( symbol,
                                           sType = 'index' )
        assetHash[ symbol ] = val

    print( 'Done with getting assetHash! Took %0.2f seconds!' % ( time.time() - t0 ) )

print( assetHash )

# ***********************************************************************                                                                   
# Get options chain
# ***********************************************************************

options = []

for symbol in ETFs:
    
    print( 'Getting options for %s...' % symbol )
    
    tmpList = utl.getOptionsChain( symbol,
                                   minExprDate  = pd.to_datetime( curDate ) + datetime.timedelta( days = 2 ),
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
                        maxPriceC   = 2000.0,
                        maxPriceA   = 4000.0,
                        minProb     = 0.75,
                        rfiDaily    = 0.0,
                        tradeFee    = 0.0,
                        nDayTimes   = 1140,
                        logFileName = None,                    
                        verbose     = 1          )                        

print( 'Found %d eligible contracts..' % len( prtObj.sortOptions( options ) ) )
actDf = prtObj.getActionDf( cash, options )

print( actDf )

actDf.to_csv( 'actDf.csv', index = False )
