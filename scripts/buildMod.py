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
minTrnDate  = '2015-01-01'
maxTrnDate  = '2015-03-31'
maxOosDate  = '2015-04-30'

velNames    = [ 'ES', 'NQ', 'US' ]

modFileName = 'model_' + minTrnDate + '_' + maxTrnDate + '.dill'

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
