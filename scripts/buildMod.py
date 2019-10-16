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

mode        = 'intraday'
diffFlag    = False
dataFlag    = False
quandlDir   = '/Users/babak/workarea/data/quandl_data'
piDir       = '/Users/babak/workarea/data/pitrading_data'

if mode == 'day':
    dfFile  = 'data/dfFile.pkl'
else:
    dfFile  = 'data/dfFile_2017plus.pkl'

minTrnDate  = pd.to_datetime( '2018-10-01 09:00:00' )
maxTrnDate  = pd.to_datetime( '2018-12-31 09:00:00' )
maxOosDate  = pd.to_datetime( '2019-07-31 23:59:00' )

indices     = [ 'INDU', 'NDX', 'SPX', 'COMPX', 'RUT',  'OEX',  
                'MID',  'SOX', 'RUI', 'RUA',   'TRAN', 'HGX',  
                'TYX',  'HUI', 'XAU'                       ] 

futures     = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

recentETFs  = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH', 
                'SMH', 'XLE', 'XLF', 'XLU', 'EWJ'          ]

ETFs        = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'GDX', 
                'OIH', 'RSX', 'SMH', 'XLE', 'XLF', 'XLV', 
                'XLU', 'FXI', 'TLT', 'EEM', 'EWJ', 'IYR', 
                'SDS', 'SLV', 'GLD', 'USO', 'UNG', 'TNA', 
                'TZA', 'FAS'                               ]

stocks      = [ 'MMM',  'AXP', 'AAPL', 'BA', 'CAT',  'CVX',
                'CSCO', 'KO',  'XOM',  'GS',  'HD',  'INTC',
                'IBM', 'JNJ',  'JPM',  'MCD', 'MRK', 'MSFT', 
                'NKE', 'PFE',  'PG',   'TRV', 'UTX', 'UNH', 
                'VZ',  'WMT',  'WBA', 'DIS'                ]

forex       = [ 'USDJPY', 'USDCHF', 'USDCAD', 'NZDUSD',
                'GBPUSD', 'EURUSD', 'AUDUSD'               ]

velNames    = ETFs + indices + futures

if diffFlag:
    nDims = len( velNames )
    for m in range( nDims ):
        velNames[m] = velNames[m] + '_Diff'
    pType = 'var'
else:
    pType = 'vel'

if diffFlag:
    modFileName = 'models/model_diff.dill'
else:
    modFileName = 'models/model.dill'

if diffFlag:
    factor = 1.0e-6
else:
    if mode == 'day':
        factor = 1.0e-2
    else:
        factor = 1.0e-4
    
# ***********************************************************************
# Get data and save to pickle file
# ***********************************************************************

if dataFlag:
    df = getDf( quandlDir, piDir, velNames )
    df = df[ df[ 'Date' ] >= minTrnDate ]
    df.to_pickle( dfFile )

# ***********************************************************************
# Build model
# ***********************************************************************

mfdMod = MfdMod(    dfFile       = dfFile,
                    minTrnDate   = minTrnDate,
                    maxTrnDate   = maxTrnDate,
                    maxOosDate   = maxOosDate,
                    velNames     = velNames,
                    maxOptItrs   = 1000,
                    optGTol      = 2.0e-2,
                    optFTol      = 2.0e-2,
                    factor       = factor,
                    regCoef      = 1.0e-5,
                    mode         = mode,
                    verbose      = 1          )

validFlag = mfdMod.build()

print( 'Success :', validFlag )

mfdMod.save( modFileName )

mfdMod.ecoMfd.pltResults( rType = 'trn', pType = pType )
mfdMod.ecoMfd.pltResults( rType = 'oos', pType = pType )

