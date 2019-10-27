# ***********************************************************************
# Import libraries
# ***********************************************************************

import os, sys, dill
import datetime
import random
import numpy as np
import pandas as pd

sys.path.append( os.path.abspath( '../' ) )

from mod.mfdMod import MfdMod
from prt.prt import MfdPrt 

# ***********************************************************************
# Set assets
# ***********************************************************************

ETFs        = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH', 
                'SMH', 'XLE', 'XLF', 'XLU', 'EWJ'          ]
cryptos     = [ 'BTC', 'ETH', 'LTC', 'ZEC' ]

# ***********************************************************************
# Input
# ***********************************************************************

assets      = cryptos
nSamples    = None
modDir      = 'crypto_models'
outFile     = 'trend_prd_success_crypto.csv'
tol         = 10.0

# ***********************************************************************
# Set some parameters
# ***********************************************************************

modFiles    = []
for item in os.listdir( modDir ):

    if item.split( '_' )[0] != 'model':
        continue

    modFiles.append( item )

if nSamples is not None:
    modFiles = random.sample( modFiles, nSamples )

totAssetVal = 1000000.0
tradeFee    = 6.95
nPrdDays    = 1
nPrdTimes   = nPrdDays * 19 * 60

# ***********************************************************************
# Evaluate
# ***********************************************************************

dateList  = []
assetList = []
valList   = []

for item in modFiles:

    if item.split( '_' )[0] != 'model':
        continue
    
    modFilePath = os.path.join( modDir, item )

    try:
        mfdPrt = MfdPrt( modFile      = modFilePath,
                         assets       = assets,
                         nPrdTimes    = nPrdTimes,
                         totAssetVal  = totAssetVal, 
                         tradeFee     = tradeFee,
                         strategy     = 'mad',
                         minProbLong  = 0.5,
                         minProbShort = 0.5,
                         vType        = 'vel',
                         verbose      = 1          )    
    except:
        continue

    trendHash  = mfdPrt.trendHash
    quoteHash  = mfdPrt.quoteHash

    ecoMfd     = mfdPrt.ecoMfd
    snapDate   = ecoMfd.maxOosDate
    endDate    = snapDate + datetime.timedelta( minutes = nPrdTimes )
    dfFile     = ecoMfd.dfFile
    df         = pd.read_pickle( dfFile ) 
    tmpDf      = df[ df.Date >= snapDate ]
    tmpDf      = tmpDf[ tmpDf.Date <= endDate ]
    tmpDf      = tmpDf.sort_values( [ 'Date' ] )

    for asset in assets:
        tmpVec = np.array( tmpDf[ asset ] )
        trend  = trendHash[ asset ][0]

        if len( tmpVec ) == 0:
            continue

        if abs( quoteHash[ asset ] - tmpVec[0] ) > tol:
            print( 'Inconsistent quote price for %s model %s; %0.2f vs %0.2f' \
            % ( asset, item, quoteHash[ asset ], tmpVec[0] ) )
            continue

        fct    = trend * ( tmpVec[-1] - tmpVec[0] )

        if fct == 0:
            continue

        val = max( 0.0, fct / abs( fct ) )

        dateList.append( snapDate )
        assetList.append( asset )
        valList.append( val )

outDf = pd.DataFrame( { 'Date'     : dateList,
                        'Asset'    : assetList,
                        'Success'  : valList    }    )

outDf.to_csv( outFile, index = False )
