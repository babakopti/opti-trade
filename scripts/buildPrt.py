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
# Get some info 
# ***********************************************************************

totAssetVal = 1000000
tradeFee    = 6.99
curDate     = ecoMfd.maxTrnDate
endDate     = ecoMfd.maxOosDate
nDays       = ( endDate - curDate ).days
nPrdTimes   = int( nDays * 8 * 60 )

# ***********************************************************************
# Build a portfolio
# ***********************************************************************

mfdPrt = MfdPrt(    modFile      = modFile,
                    assets       = assets,
                    nPrdTimes    = nPrdTimes,
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
