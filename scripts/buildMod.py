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
dfFile      = 'data/dfFile.pkl'
minTrnDate  = pd.to_datetime( '2015-01-01 00:00:00' )
maxTrnDate  = pd.to_datetime( '2015-01-31 23:59:00' )
maxOosDate  = pd.to_datetime( '2015-02-15 23:59:00' )

velNames    = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

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
                    maxOptItrs   = 2000,
                    optGTol      = 1.0e-2,
                    optFTol      = 1.0e-2,
                    regCoef      = 1.0e-4,
                    minMerit     = 0.65,
                    maxBias      = 0.10,
                    varFiltFlag  = False,
                    validFlag    = False,
                    smoothCount  = None,
                    verbose      = 1          )

validFlag = mfdMod.build()

print( 'Success :', validFlag )

mfdMod.save( modFileName )

mfdMod.ecoMfd.pltResults( rType = 'trn', pType = 'vel' )
