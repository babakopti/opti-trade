# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import datetime
import numpy as np
import pandas as pd

from utils import getDf

sys.path.append( os.path.abspath( '../' ) )

from mod.mfdMod import MfdMod

# ***********************************************************************
# Set some parameters and read data
# ***********************************************************************

dataFlag    = False
quandlDir   = '/Users/babak/workarea/data/quandl_data'
piDir       = '/Users/babak/workarea/data/pitrading_data'
dfFile      = 'data/light_dfFile.pkl'
minTrnDate  = pd.to_datetime( '2014-01-01 00:00:00' )
maxTrnDate  = pd.to_datetime( '2018-12-31 23:59:00' )
maxOosDate  = pd.to_datetime( '2019-08-02 23:59:00' )

indices     = [ 'INDU', 'NDX', 'SPX', 'COMPX', 'RUT',  'OEX',  
                'MID',  'SOX', 'RUI', 'RUA',   'TRAN', 'HGX',  
                'TYX',  'HUI', 'XAU'                       ] 

futures     = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

ETFs        = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH', 
                'SMH', 'XLE', 'XLF', 'XLU', 'EWJ'          ]

stocks      = [ 'MMM',  'AXP', 'AAPL', 'BA', 'CAT',  'CVX',
                'CSCO', 'KO',  'XOM',  'GS',  'HD',  'INTC',
                'IBM', 'JNJ',  'JPM',  'MCD', 'MRK', 'MSFT', 
                'NKE', 'PFE',  'PG',   'TRV', 'UTX', 'UNH', 
                'VZ',  'WMT',  'WBA', 'DIS'                ]
forex       = [ 'USDJPY', 'USDCHF', 'USDCAD', 'NZDUSD',
                'GBPUSD', 'EURUSD', 'AUDUSD'               ]

velNames    = indices + futures + ETFs #+ stocks

modFileName = 'model.dill'

# ***********************************************************************
# Get data and save to pickle file
# ***********************************************************************

if dataFlag:
    df = getDf( quandlDir, piDir, velNames )
    df.to_pickle( dfFile )

# ***********************************************************************
# Build model
# ***********************************************************************

mfdMod = MfdMod(    dfFile       = dfFile,
                    minTrnDate   = minTrnDate,
                    maxTrnDate   = maxTrnDate,
                    maxOosDate   = maxOosDate,
                    velNames     = velNames,
                    maxOptItrs   = 50,
                    optGTol      = 1.0e-2,
                    optFTol      = 1.0e-2,
                    regCoef      = 1.0e-5,
                    minMerit     = 0.65,
                    maxBias      = 0.10,
                    varFiltFlag  = False,
                    validFlag    = False,
                    smoothCount  = None,
                    verbose      = 1          )

validFlag = mfdMod.build()

print( 'Success :', validFlag )

mfdMod.save( modFileName )

mfdMod.ecoMfd.pltResults( rType = 'all', pType = 'vel' )
