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

from dat.assets import INDEXES
from mod.mfdMod import MfdMod

# ***********************************************************************
# Set some parameters and read data
# ***********************************************************************

cryptoDir   = '/Users/babak/workarea/data/crypto_data'
piDir       = '/Users/babak/workarea/data/pitrading_data'

dfFile      = 'data/dfFile_crypto.pkl'

minTrnDate  = pd.to_datetime( '2019-09-15 00:00:00' )
maxTrnDate  = pd.to_datetime( '2020-09-14 23:59:00' )
maxOosDate  = pd.to_datetime( '2020-09-17 23:59:00' )

cryptos     = [ 'BTC', 'ETH', 'LTC', 'ZEC' ]
velNames    = INDEXES + cryptos + [ 'VIX' ]

modFileName = 'models/crypto_model.dill'

selParams = {
    'inVelNames': cryptos, 
    'maxNumVars': 10,
    'minImprov': 0.005,
    'strategy': 'forward',
}

# ***********************************************************************
# Build model
# ***********************************************************************

mfdMod = MfdMod(    dfFile       = dfFile,
                    minTrnDate   = minTrnDate,
                    maxTrnDate   = maxTrnDate,
                    maxOosDate   = maxOosDate,
                    velNames     = velNames,
                    maxOptItrs   = 500,
                    optGTol      = 1.0e-5,
                    optFTol      = 1.0e-5,
                    regCoef      = 5.0e-3,
                    factor       = 4.0e-05,
                    selParams    = selParams,                    
                    logFileName  = None,                    
                    verbose      = 1          )

validFlag = mfdMod.build()

print( 'Success :', validFlag )

mfdMod.save( modFileName )

mfdMod.ecoMfd.pltResults( rType = 'all', pType = 'vel' )

