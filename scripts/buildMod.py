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
from dat.assets import ETF_HASH, FUTURES

# ***********************************************************************
# Set some parameters and read data
# ***********************************************************************

dfFile      = 'data/dfFile_2021-02-01 09:30:04.pkl'

minTrnDate  = pd.to_datetime( '2019-10-31 09:00:00' )
maxTrnDate  = pd.to_datetime( '2020-10-30 09:00:00' )
maxOosDate  = pd.to_datetime( '2020-12-15 23:59:00' )

velNames    = list( ETF_HASH.keys() ) + FUTURES

pType       = 'vel'

modFileName = 'models/model.dill'

# ***********************************************************************
# Build model
# ***********************************************************************

mfdMod = MfdMod(    dfFile       = dfFile,
                    minTrnDate   = minTrnDate,
                    maxTrnDate   = maxTrnDate,
                    maxOosDate   = maxOosDate,
                    velNames     = velNames,
                    maxOptItrs   = 100,
                    optGTol      = 1.0e-6,
                    optFTol      = 1.0e-6,
                    factor       = 5.0e-2,
                    regCoef      = 0.0,
                    smoothCount  = None,
                    logFileName  = None,
                    mode         = 'day',
                    verbose      = 1          )
validFlag = mfdMod.build()

print( 'Success :', validFlag )

mfdMod.save( modFileName )
mfdMod.ecoMfd.pltResults( rType = 'trn', pType = pType )
#mfdMod.ecoMfd.pltResults( rType = 'oos', pType = pType )

