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
    dfFile  = 'data/dfFile_daily.pkl'
else:
    dfFile  = 'data/dfFile_kibot_2016plus.pkl'

minTrnDate  = pd.to_datetime( '2018-01-23 09:00:00' )
maxTrnDate  = pd.to_datetime( '2019-01-18 09:00:00' )
maxOosDate  = pd.to_datetime( '2019-01-21 09:00:00' )

indices     = [ 'INDU', 'NDX', 'SPX', 'COMPQ', 'RUT',  'OEX',  
                'MID',  'SOX', 'RUI', 'RUA',   'TRAN', 'HGX',  
                'TYX',  'XAU'                      ] 

oldETFs     = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH', 
                'SMH', 'XLF', 'EWJ'          ]
ETFs        = [ 'TQQQ', 'SPY', 'DDM', 'MVV', 'UWM', 'DIG', 'USD',
                'ERX',  'UYG', 'UPW', 'UGL', 'BIB', 'UST', 'UBT'  ]

ETFs        = ETFs + oldETFs
ETFs        = list( set( ETFs ) )

futures     = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

allETFs     = [ 'TQQQ', 'SPY', 'DDM', 'MVV', 'UWM',  'SAA',
                'UYM',  'UGE', 'UCC', 'FINU', 'RXL', 'UXI',
                'URE',  'ROM', 'UJB', 'AGQ',  'DIG', 'USD',
                'ERX',  'UYG', 'UCO', 'BOIL', 'UPW', 'UGL',
                'BIB', 'UST', 'UBT'  ]

velNames    = allETFs + futures 

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
        factor = 4.0e-5
    
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
                    maxOptItrs   = 300,
                    optGTol      = 5.0e-2,
                    optFTol      = 5.0e-2,
                    factor       = factor,
                    regCoef      = 1.0e-3,
                    smoothCount  = None,
                    verbose      = 1          )

validFlag = mfdMod.build()

print( 'Success :', validFlag )

mfdMod.save( modFileName )

mfdMod.ecoMfd.pltResults( rType = 'trn', pType = pType )
mfdMod.ecoMfd.pltResults( rType = 'oos', pType = pType )

