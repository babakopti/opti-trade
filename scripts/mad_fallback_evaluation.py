# ***********************************************************************
# Import libraries
# ***********************************************************************

import os
import sys
import dill
import datetime
import numpy as np
import pandas as pd

sys.path.append( os.path.abspath( '../' ) )

from prt.prt import MfdPrt

# ***********************************************************************
# Set some parameters
# ***********************************************************************

fallBacks   = [ None, 'sign_trick', 'macd', 'msd', 'zero' ]

modDir      = 'models'

ETFs        = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH', 
                'SMH', 'XLE', 'XLF', 'XLU', 'EWJ'          ]
assets      = ETFs
nPrdDays    = 1
dfFile      = 'data/dfFile_2017plus.pkl'

outFile     = 'fallBack.csv'

# ***********************************************************************
# Some utility functions
# ***********************************************************************

def getNumPrdTimes( curDate, df ):

    begDate     = pd.to_datetime( curDate )
    endDate     = begDate + datetime.timedelta( days = nPrdDays )
    tmpDf       = df[ df.Date >= begDate ]
    tmpDf       = tmpDf[ tmpDf.Date < endDate ]
    nPrdTimes   = tmpDf.shape[0]

    return nPrdTimes

def getActTrends( curDate, df ):

    begDate = pd.to_datetime( curDate )
    endDate = begDate + datetime.timedelta( days = nPrdDays )

    df = df[ df.Date >= begDate ]
    df = df[ df.Date < endDate ]
    df = df.sort_values( [ 'Date' ] )

    trendHash = {}

    for asset in assets:
        tmpVec = np.array( df[ asset ] )

        if len( tmpVec ) == 0:
            trend = 0
            print( 'Warning: no data found for', curDate, asset )
        else:
            nPrdTimes = getNumPrdTimes( curDate, df )

            assert len( tmpVec ) == nPrdTimes, 'Incosistent size!'

            curPrice = tmpVec[0]
            trend    = 0.0

            for i in range( nPrdTimes ):

                prdPrice = tmpVec[i]

                if prdPrice > curPrice:
                    fct = 1.0
                else:
                    fct = -1.0

                trend += fct

            trend /= nPrdTimes

        trendHash[ asset ] = trend

    return trendHash

# ***********************************************************************
# Analyze and Evaluate
# ***********************************************************************

df          = pd.read_pickle( dfFile )
cntList     = []
successList = []
rateList    = []

for fallBack in fallBacks:

    totCnt  = 0
    success = 0
    
    for fileName in os.listdir( modDir ):

        tmpList  = os.path.splitext( fileName )
        
        if tmpList[1] != '.dill':
            continue

        baseName = tmpList[0]
        tmpList  = baseName.split( '_' )

        if tmpList[0] != 'model':
            continue

        curDate     = pd.to_datetime( tmpList[1] )
        modFilePath = os.path.join( modDir, fileName )
        nPrdTimes   = getNumPrdTimes( curDate, df )

        try:
            mfdPrt = MfdPrt( modFile      = modFilePath,
                             assets       = assets,
                             nRetTimes    = 360,
                             nPrdTimes    = nPrdTimes,
                             totAssetVal  = 1000000, 
                             tradeFee     = 6.95,
                             strategy     = 'mad',
                             minProbLong  = 0.5,
                             minProbShort = 0.5,
                             vType        = 'vel',
                             fallBack     = fallBack,
                             optTol       = 1.0e-6,
                             verbose      = 1          )
    
            trendHash = mfdPrt.trendHash
        except:
            continue

        actTrendHash = getActTrends( curDate, df )

        for asset in assets:
            totCnt += 1

            if trendHash[ asset ][0] * actTrendHash[ asset ] > 0:
                success += 1

    cntList.append( totCnt )
    successList.append( success )
    rateList.append( round( success / float( totCnt ), 3 ) )

outDf = pd.DataFrame( { 'FallBack' : fallBacks,
                        'Total'    : cntList,
                        'Success'  : successList,
                        'Rate'     : rateList     } )

print( outDf )

outDf.to_csv( outFile, index = False )
