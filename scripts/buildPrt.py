# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import datetime
import dill
import numpy as np
import pandas as pd

from utils import getDf

sys.path.append( os.path.abspath( '../' ) )
sys.path.append( os.path.abspath( '../../etrade-api-wrapper' ) )

from prt.prt import MfdPrt
from mod.mfdMod import MfdMod

from etrade import Etrade

# ***********************************************************************
# Set some parameters and read data
# ***********************************************************************

indices     = [ 'INDU', 'NDX', 'SPX', 'COMPX', 'RUT',  'OEX',  
                'MID',  'SOX', 'RUI', 'RUA',   'TRAN', 'HGX',  
                'TYX',  'HUI', 'XAU'                       ] 

ETFs        = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH', 
                'SMH', 'XLE', 'XLF', 'XLU', 'EWJ'          ]

assets      = ETFs 

modFile     = 'model_2018-01-03 09:00:00.dill'

# ***********************************************************************
# Read model file
# ***********************************************************************

mfdMod      = dill.load( open( modFile, 'rb' ) )
ecoMfd      = mfdMod.ecoMfd

# ***********************************************************************
# Get quotes and other info 
# ***********************************************************************

if False:
    etrade = Etrade( 'configs/config.ini', sandBox = False )

    totAssetVal = etrade.getTotalValue()
    tradeFee    = 6.95
    quoteHash   = {}

    for asset in assets:
        price = etrade.getQuote( asset ) 

        if price is None:
            print( 'Skipping', asset, '...' )
            continue

        quoteHash[ asset ] = etrade.getQuote( asset )

else:
    totAssetVal = 1000000.0
    tradeFee    = 6.95
    quoteHash   = {}

    for asset in assets:
        
        if asset in [ 'INDU', 'COMPX', 'TRAN' ]:
            continue

        for m in range( ecoMfd.nDims ):
            if ecoMfd.velNames[m] == asset:
                break

        assert m < ecoMfd.nDims, \
            'Asset %s not found in the model!' % asset

        tmp       = ecoMfd.deNormHash[ asset ]
        slope     = tmp[0]
        intercept = tmp[1]
        price     = slope * ecoMfd.actOosSol[m][-1] + intercept
        
        quoteHash[ asset ] = price

print( quoteHash )

# ***********************************************************************
# Build a portfolio
# ***********************************************************************

mfdPrt = MfdPrt(    modFile      = modFile,
                    curDate      = ecoMfd.maxTrnDate.strftime( '%Y-%m-%d' ),
                    endDate      = ecoMfd.maxOosDate.strftime( '%Y-%m-%d' ),
                    assets       = list( quoteHash.keys() ),
                    quoteHash    = quoteHash,
                    totAssetVal  = totAssetVal, 
                    tradeFee     = tradeFee,
                    strategy     = 'mad',
                    minProbLong  = 0.5,
                    minProbShort = 0.5,
                    verbose      = 1          )

#mfdMod = dill.load( open( modFile, 'rb' ) )
#mfdMod.ecoMfd.pltResults()
print(mfdPrt.trendHash)
print(mfdPrt.getPortfolio())
mfdPrt.pltIters()
