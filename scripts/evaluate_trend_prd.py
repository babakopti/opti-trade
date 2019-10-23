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

modDir      = '/Volumes/Public/workarea/opti-trade/scripts/models_daily_20191020'

modFiles    = os.listdir( modDir )

ETFs        = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH', 
                'SMH', 'XLE', 'XLF', 'XLU', 'EWJ'          ]

assets      = ETFs
totAssetVal = 1000000.0
tradeFee    = 6.95
nPrdDays    = 1
nPrdTimes   = nPrdDays * 19 * 60

# ***********************************************************************
# Import libraries
# ***********************************************************************

dateList  = []
assetList = []
valList   = []

for item in modFiles:

    if item.split( '_' )[0] != 'model':
        continue
    
    modFilePath = os.path.join( modDir, item )

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

        assert abs( quoteHash[ asset ] - tmpVec[0] ) < 0.01, \
            'Inconsistent quote price for %s model %s; %0.2f vs %0.2f' \
            % ( asset, item, quoteHash[ asset ], tmpVec[0] )

        fct    = trend * ( np.mean( tmpVec ) - tmpVec[0] )
        
        if fct != 0:
            val = fct / abs( fct )
        else:
            val = 0.0

        dateList.append( snapDate )
        assetList.append( asset )
        valList.append( val )

outDf = pd.DataFrame( { 'Date'     : dateList,
                        'Asset'    : assetList,
                        'Success'  : valList    }    )

outDf.to_csv( 'trend_prd_success.csv', index = False )
