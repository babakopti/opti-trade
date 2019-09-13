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

assets      = indices + ETFs 

modFile     = 'models/model_May_2019.dill'

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
    totAssetVal = 500000.0
    tradeFee    = 6.95
    quoteHash = { 'NDX': 7887.581, 
                  'SPX': 3000.93, 
                  'RUT': 1575.7122, 
                  'OEX': 1327.87, 
                  'MID': 1964.11, 
                  'SOX': 1606.19, 
                  'RUI': 1658.7848, 
                  'RUA': 1761.0648, 
                  'HGX': 334.97, 
                  'TYX': 22.08, 
                  'HUI': 211.24, 
                  'XAU': 92.35, 
                  'QQQ': 192.43, 
                  'SPY': 300.25, 
                  'DIA': 271.69, 
                  'MDY': 358.48, 
                  'IWM': 157.0, 
                  'OIH': 13.15, 
                  'SMH': 121.78, 
                  'XLE': 60.98, 
                  'XLF': 28.12, 
                  'XLU': 63.2, 
                  'EWJ': 55.84   }
    
print( quoteHash )

# ***********************************************************************
# Build a portfolio
# ***********************************************************************

mfdPrt = MfdPrt(    modFile      = modFile,
                    curDate      = '2019-05-25',
                    endDate      = '2019-06-01', 
                    assets       = list( quoteHash.keys() ),
                    quoteHash    = quoteHash,
                    totAssetVal  = totAssetVal, 
                    tradeFee     = tradeFee,
                    minGainRate  = 0.0,
                    minProbLong  = 0.75,
                    minProbShort = 0.75,
                    verbose      = 1          )

#print( mfdPrt.getPrdTrends() )

#mfdMod = dill.load( open( modFile, 'rb' ) )
#mfdMod.ecoMfd.pltResults()

mfdPrt.getPortfolio()
