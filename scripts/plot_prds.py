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

assets      = ETFs
modDir      = '/Volumes/Public/workarea/opti-trade/scripts/models_daily_20191020'
dfFile      = 'data/dfFile_2017plus.pkl'
outFile     = 'model_prds.csv'

# ***********************************************************************
# Set some parameters
# ***********************************************************************

modFiles    = []
for item in os.listdir( modDir ):

    if item.split( '_' )[0] != 'model':
        continue

    modFiles.append( item )

modFiles.sort()

totAssetVal = 1000000.0
tradeFee    = 6.95
nPrdDays    = 1

# ***********************************************************************
# Evaluate
# ***********************************************************************

dataDf    = pd.read_pickle( dfFile )
dataDf    = dataDf[ [ 'Date' ] + assets ]
dataDf    = dataDf.sort_values( 'Date' )
dataDf    = dataDf.drop_duplicates()
dataDf    = dataDf.dropna()
dataDf    = dataDf.reset_index( drop = True )
outDf     = pd.DataFrame()

for item in modFiles:

    if item.split( '_' )[0] != 'model':
        continue

    print( 'Processing', item )

    baseName = os.path.splitext( item )[0]
    snapDate = pd.to_datetime( baseName.split( '_' )[1] )
    endDate  = snapDate + datetime.timedelta( days = nPrdDays )
    tmpDf    = dataDf[ dataDf.Date >= snapDate ]
    tmpDf    = tmpDf[ tmpDf.Date < endDate ]

    nPrdTimes   = tmpDf.shape[0]
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

    ecoMfd     = mfdPrt.ecoMfd
    velNames   = ecoMfd.velNames
    prdSol     = mfdPrt.prdSol
    stdVec     = mfdPrt.stdVec

    assert snapDate == ecoMfd.maxOosDate, \
        'Iconsistent snapDate!'
    
    for m in range( ecoMfd.nDims ):

        if velNames[m] not in assets:
            continue

        prdList = []
        for i in range( nPrdTimes ):
            prdList.append( prdSol[m][i] )

        tmpDf[ velNames[m] + '_prd' ] = prdList
        tmpDf[ velNames[m] + '_std' ] = stdVec[m]

    outDf = pd.concat( [ outDf, tmpDf ] )

outDf = outDf.sort_values( 'Date' )
outDf = outDf.reset_index( drop = True )

outDf.to_csv( outFile, index = False )
