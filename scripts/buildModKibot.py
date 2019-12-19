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

dfFile  = 'data/dfFile_kibot.pkl'

minTrnDate  = pd.to_datetime( '2017-03-01 09:00:00' )
maxTrnDate  = pd.to_datetime( '2018-03-01 09:00:00' )
maxOosDate  = pd.to_datetime( '2018-03-05 23:59:00' )

indexes     = [ 'INDU', 'NDX', 'SPX', 'COMPQ', 'RUT',  'OEX',  
                'MID',  'SOX', 'RUI', 'RUA',   'TRAN', 'HGX',  
                'TYX',  'XAU'                      ] 

ETFs        = [ 'TQQQ', 'SPY', 'DDM', 'MVV', 'UWM', 'DIG', 'USD',
                'ERX',  'UYG', 'UPW', 'UGL', 'BIB', 'UST', 'UBT'  ]

#ETFs        = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH', 
#                'SMH', 'XLE', 'XLF', 'XLU', 'EWJ'          ]

futures     = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

velNames    = ETFs + futures + indexes

pType = 'vel'

modFileName = 'models/model_kibot.dill'

factor = 4.0e-5
    
# ***********************************************************************
# Build model
# ***********************************************************************

mfdMod = MfdMod(    dfFile       = dfFile,
                    minTrnDate   = minTrnDate,
                    maxTrnDate   = maxTrnDate,
                    maxOosDate   = maxOosDate,
                    velNames     = velNames,
                    maxOptItrs   = 200,
                    optGTol      = 5.0e-2,
                    optFTol      = 5.0e-2,
                    factor       = factor,
                    regCoef      = 1.0e-3,
                    selParams    = None,
                    atnFct       = 1.0,
                    verbose      = 1          )

validFlag = mfdMod.build()

print( 'Success :', validFlag )

mfdMod.save( modFileName )

mfdMod.ecoMfd.pltResults( rType = 'trn', pType = pType )
mfdMod.ecoMfd.pltResults( rType = 'oos', pType = pType )

