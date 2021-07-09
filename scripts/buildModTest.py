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

sys.path.append( os.path.abspath( '..' ) )

from mod.mfdMod import MfdMod
from dat.assets import ETF_HASH, FUTURES

# ***********************************************************************
# Set some parameters and read data
# ***********************************************************************

dfFile      = 'data/dfFile_2021-02-01 09:30:04.pkl' #'data/dfFile_returns_logs.pkl'

minTrnDate  = pd.to_datetime( '2020-02-05 09:00:00' )
maxTrnDate  = pd.to_datetime( '2020-12-31 09:00:00' )
maxOosDate  = pd.to_datetime( '2021-02-01 23:59:00' )

velNames    = [
    "SPY",
    "MVV",
    "AGQ",
    "BOIL",
    "UST",
]

# ***********************************************************************
# Build model
# ***********************************************************************

mfdMod = MfdMod(    dfFile       = dfFile,
                    minTrnDate   = minTrnDate,
                    maxTrnDate   = maxTrnDate,
                    maxOosDate   = maxOosDate,
                    velNames     = velNames,
                    optType      = 'SLSQP',
                    maxOptItrs   = 300,
                    optGTol      = 1.0e-8,
                    optFTol      = 1.0e-8,
                    factor       = 1.0,
                    regCoef      = 1.0e-6,
                    smoothCount  = None,
                    logFileName  = None,
                    mode         = 'day',
                    verbose      = 1          )
validFlag = mfdMod.build()

print( 'Success :', validFlag )

nGammaVec = mfdMod.ecoMfd.nParams - mfdMod.ecoMfd.nDims
GammaVec =  mfdMod.ecoMfd.params[:nGammaVec]

print("Final Max Gamma:", max(GammaVec))
print("Final Min Gamma:", min(GammaVec))

#mfdMod.ecoMfd.pltResults( rType = 'trn', pType = pType )
mfdMod.ecoMfd.pltResults( rType = 'all', pType = "vel" )

