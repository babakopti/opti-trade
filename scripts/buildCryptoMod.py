# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import datetime
import numpy as np
import pandas as pd

from utils import getCryptoDf

sys.path.append( os.path.abspath( '../' ) )

from mod.mfdMod import MfdMod

# ***********************************************************************
# Set some parameters and read data
# ***********************************************************************

dataFlag    = True
cryptoDir   = '/Users/babak/workarea/data/crypto_data'
piDir       = '/Users/babak/workarea/data/pitrading_data'

dfFile      = 'data/dfFile_crypto.pkl'

minTrnDate  = pd.to_datetime( '2018-01-01 00:00:00' )
maxTrnDate  = pd.to_datetime( '2019-06-30 23:59:00' )
maxOosDate  = pd.to_datetime( '2019-07-10 23:59:00' )

cryptos     = [ 'BTC', 'ETH', 'LTC', 'ZEC' ]
indices     = [ 'INDU', 'NDX', 'SPX', 'COMPX', 'RUT',  'OEX',  
                'MID',  'SOX', 'RUI', 'RUA',   'TRAN', 'HGX',  
                'TYX',  'HUI', 'XAU'                       ] 
forex       = [ 'USDJPY', 'USDCHF', 'USDCAD', 'NZDUSD',
                'GBPUSD', 'EURUSD', 'AUDUSD'               ]

velNames    = indices + cryptos

modFileName = 'models/crypto_model.dill'

factor      = 2.0e-5
    
# ***********************************************************************
# Get data and save to pickle file
# ***********************************************************************

if dataFlag:
    df = getCryptoDf( cryptoDir, piDir, velNames )
    df = df[ df.Date >= minTrnDate ]
    df.to_pickle( dfFile )
sys.exit()
# ***********************************************************************
# Build model
# ***********************************************************************

mfdMod = MfdMod(    dfFile       = dfFile,
                    minTrnDate   = minTrnDate,
                    maxTrnDate   = maxTrnDate,
                    maxOosDate   = maxOosDate,
                    velNames     = velNames,
                    maxOptItrs   = 100,
                    optGTol      = 2.0e-2,
                    optFTol      = 2.0e-2,
                    factor       = factor,
                    regCoef      = 1.0e-5,
                    mode         = 'intraday',
                    verbose      = 1          )

validFlag = mfdMod.build()

print( 'Success :', validFlag )

mfdMod.save( modFileName )

mfdMod.ecoMfd.pltResults( rType = 'all', pType = 'vel' )

