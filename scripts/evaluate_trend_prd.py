# ***********************************************************************
# Import libraries
# ***********************************************************************

import os, sys, dill
import datetime
import numpy as np
import pandas as pd

sys.path.append( os.path.abspath( '../' ) )

from mod.mfdMod import MfdMod
from prt.prt import MfdPrt 

# ***********************************************************************
# Import libraries
# ***********************************************************************

modFiles    = os.listdir( 'models' )

ETFs        = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH', 
                'SMH', 'XLE', 'XLF', 'XLU', 'EWJ'          ]

assets      = ETFs
totAssetVal = 1000000.0
tradeFee    = 6.95
nPrdDays    = 7
nPrdTimes   = nPrdDays * 19 * 60

# ***********************************************************************
# Import libraries
# ***********************************************************************

totCnt   = 0
matchCnt = 0

for item in modFiles:

    if item.split( '_' )[0] != 'model':
        continue
    
    modFilePath = os.path.join( 'models', item )

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

    trendHash  = mfdPrt.trendHash
    quoteHash  = mfdPrt.quoteHash

    mfdMod     = dill.load( open( modFilePath, 'rb' ) )
    ecoMfd     = mfdMod.ecoMfd
    maxOosDate = ecoMfd.maxOosDate
    endDate    = maxOosDate + datetime.timedelta( days = nPrdDays )
    dfFile     = ecoMfd.dfFile
    df         = pd.read_pickle( dfFile ) 
    tmpDf      = df[ df.Date >= maxOosDate ]
    tmpDf      = tmpDf[ tmpDf.Date <= endDate ]
    tmpDf      = tmpDf.sort_values( [ 'Date' ] )

    for asset in assets:
        tmpVec = np.array( tmpDf[ asset ] )
        trend  = trendHash[ asset ][0]

        assert abs( quoteHash[ asset ] - tmpVec[0] ) < 0.01, \
            'Inconsistent quote price for %s model %s; %0.2f vs %0.2f' \
            % ( asset, item, quoteHash[ asset ], tmpVec[0] )

        fct    = trend * ( np.mean( tmpVec ) - tmpVec[0] )
        
        totCnt += 1
        
        if fct > 0:
            matchCnt += 1
    
print( '%d out of %d matched trend! That is %0.3f ratio' \
           % ( matchCnt, totCnt, round( float( matchCnt ) / totCnt, 3 ) ) )
