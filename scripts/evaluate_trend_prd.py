# ***********************************************************************
# Import libraries
# ***********************************************************************

import os, sys, dill
import datetime
import random
import talib
import numpy as np
import pandas as pd

sys.path.append( os.path.abspath( '../' ) )

from mod.mfdMod import MfdMod
from prt.prt import MfdPrt 

# ***********************************************************************
# Set assets
# ***********************************************************************

ETFs        = [ 'TQQQ', 'SPY', 'DDM', 'MVV', 'UWM', 'DIG', 'USD',
                'ERX',  'UYG', 'UPW', 'UGL', 'BIB', 'UST', 'UBT'  ]

# ***********************************************************************
# Input
# ***********************************************************************

assets      = ETFs
nSamples    = None
modDir      = 'models'
outFile     = 'trend_prd_success.csv'
sumFile     = 'summary_trend_prd_success.csv'
tol         = 0.1

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
nPrdTimes   = nPrdDays * 17 * 60

# ***********************************************************************
# Some utility functions
# ***********************************************************************

def MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9):
    '''
    Function to return the difference between the most recent 
    MACD value and MACD signal. Positive values are long
    position entry signals 

    optional args:
        fastperiod = 6
        slowperiod = 45
        signalperiod = 9

    Returns: macd - signal
    '''
    macd, signal, hist = talib.MACD(prices, 
                                    fastperiod=fastperiod, 
                                    slowperiod=slowperiod, 
                                    signalperiod=signalperiod)

    return macd[len(macd)-1] - signal[len(signal)-1]

# ***********************************************************************
# Evaluate
# ***********************************************************************

dateList  = []
assetList = []
mfdList   = []
msdList   = []
macdList  = []
hybList   = []

for item in modFiles:

    if item.split( '_' )[0] != 'model':
        continue
    
    modFilePath = os.path.join( modDir, item )

    try:
        mfdPrt = MfdPrt( modFile      = modFilePath,
                         assets       = assets,
                         nPrdTimes    = nPrdTimes,
                         nRetTimes    = 30,
                         strategy     = 'mad',
                         minProbLong  = 0.5,
                         minProbShort = 0.5,
                         vType        = 'vel',
                         fallBack     = 'macd',
                         verbose      = 1          )
    except:
        continue

    quoteHash  = mfdPrt.quoteHash
    ecoMfd     = mfdPrt.ecoMfd
    snapDate   = ecoMfd.maxOosDate
    endDate    = snapDate + datetime.timedelta( minutes = nPrdTimes )
    dfFile     = ecoMfd.dfFile
    df         = pd.read_pickle( dfFile )
    tmpDf      = df[ df.Date >= snapDate ]
    tmpDf      = tmpDf[ tmpDf.Date <= endDate ]
    tmpDf      = tmpDf.sort_values( [ 'Date' ] )

    mfdTrendHash  = mfdPrt.trendHash

    tmpDate       = snapDate - datetime.timedelta( days = 20 )
    tmpDfHistMsd  = df[ df.Date >= tmpDate ]
    tmpDfHistMsd  = tmpDfHistMsd[ tmpDfHistMsd.Date <= snapDate ]
    tmpDfHistMsd  = tmpDfHistMsd.sort_values( [ 'Date' ] )

    tmpDate       = snapDate - datetime.timedelta( days = 360 )
    tmpDfHistMacd = df[ df.Date >= tmpDate ]
    tmpDfHistMacd = tmpDfHistMacd[ tmpDfHistMacd.Date <= snapDate ]
    tmpDfHistMacd = tmpDfHistMacd.sort_values( [ 'Date' ] )

    for asset in assets:

        tmpVec = np.array( tmpDf[ asset ] )

        if len( tmpVec ) == 0:
            continue

        if abs( quoteHash[ asset ] - tmpVec[0] ) > tol:
            print( 'Inconsistent quote price for %s model %s; %0.2f vs %0.2f' \
            % ( asset, item, quoteHash[ asset ], tmpVec[0] ) )
            continue

        dateList.append( snapDate )
        assetList.append( asset )

        mfdTrend  = mfdTrendHash[ asset ][0]

        fct    = mfdTrend * ( np.mean( tmpVec ) - tmpVec[0] )

        if fct == 0:
            val = None
        else:
            val = max( 0.0, fct / abs( fct ) )

        mfdList.append( val )

        meanVal  = np.mean( tmpDfHistMsd[ asset ] )
        sigmaVal = np.std( tmpDfHistMsd[ asset ] )

        if tmpVec[0] < meanVal - 1.75 * sigmaVal:
            msdTrend = 1.0
        elif tmpVec[0] > meanVal + 1.75 * sigmaVal:
            msdTrend = -1.0
        else:
            msdTrend = 0.0

        fct    = msdTrend * ( np.mean( tmpVec ) - tmpVec[0] )

        if fct == 0:
            val = None
        else:
            val = max( 0.0, fct / abs( fct ) )

        msdList.append( val )

        macdTrend = MACD( np.array(tmpDfHistMacd[ asset ]) )

        fct    = macdTrend * ( np.mean( tmpVec ) - tmpVec[0] )

        if fct == 0:
            val = None
        else:
            val = max( 0.0, fct / abs( fct ) )

        macdList.append( val )

        if mfdTrend == 0.0:
            hybTrend = macdTrend
        else:
            hybTrend = mfdTrend

        fct    = hybTrend * ( np.mean( tmpVec ) - tmpVec[0] )

        if fct == 0:
            val = None
        else:
            val = max( 0.0, fct / abs( fct ) )

        hybList.append( val )

outDf = pd.DataFrame( { 'Date'     : dateList,
                        'Asset'    : assetList,
                        'MFD'      : mfdList,
                        'MSDEV'    : msdList,
                        'MACD'     : macdList, 
                        'Hybrid'   : hybList    }    )

outDf.to_csv( outFile, index = False )

sumDf = outDf.groupby( 'Date', as_index = False )[ 'MFD', 'MSDEV', 'MACD', 'Hybrid' ].mean()

sumDf.to_csv( sumFile, index = False )

print( 'Mean success rate:', { 'MFD'    : sumDf.MFD.mean(),
                               'MSDEV'  : sumDf.MSDEV.mean(),
                               'MACD'   : sumDf.MACD.mean(), 
                               'Hybrid' : sumDf.Hybrid.mean() } )
