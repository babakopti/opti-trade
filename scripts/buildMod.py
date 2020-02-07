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
from dat.assets import OLD_ETF_HASH

# ***********************************************************************
# Set some parameters and read data
# ***********************************************************************

dfFile      = 'data/dfFile_long_term_all.pkl'

minTrnDate  = pd.to_datetime( '2003-01-01 09:00:00' )
maxTrnDate  = pd.to_datetime( '2017-12-31 09:00:00' )
maxOosDate  = pd.to_datetime( '2020-02-04 15:00:00' )

indices     = [ 'INDU', 'NDX', 'SPX', 'RUT', 'OEX',  
                'MID',  'SOX', 'RUI', 'RUA', 'TRAN',
                'HGX',  'TYX', 'XAU'               ] 

futures     = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

ETFs        = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH', 
                'SMH', 'XLE', 'XLF', 'XLU', 'EWJ'          ]

velNames    = ETFs + indices + futures

pType       = 'vel'

modFileName = 'models/model_long_term.dill'

factor      = 1.0e-5
    
# ***********************************************************************
# Build model
# ***********************************************************************

mfdMod = MfdMod(    dfFile       = dfFile,
                    minTrnDate   = minTrnDate,
                    maxTrnDate   = maxTrnDate,
                    maxOosDate   = maxOosDate,
                    velNames     = velNames,
                    maxOptItrs   = 500,
                    optGTol      = 1.0e-2,
                    optFTol      = 1.0e-2,
                    factor       = factor,
                    regCoef      = 1.0e-3,
                    smoothCount  = None,
                    logFileName  = None,
                    verbose      = 1          )

validFlag = mfdMod.build()

print( 'Success :', validFlag )

mfdMod.save( modFileName )

#mfdMod.ecoMfd.pltResults( rType = 'trn', pType = pType )
mfdMod.ecoMfd.pltResults( rType = 'oos', pType = pType )

