# ***********************************************************************
# Import libraries
# ***********************************************************************

import os, sys, dill
import datetime
import random
import talib
import numpy as np
import pandas as pd

sys.path.append( os.path.abspath( '../' ) )

# ***********************************************************************
# Some definitions
# ***********************************************************************

modDir      = 'models'
modFiles    = []
nSamples    = None

# ***********************************************************************
# Set some parameters
# ***********************************************************************

if len( modFiles ) == 0:
    for item in os.listdir( modDir ):

        if item.split( '_' )[0] != 'model':
            continue

        modFiles.append( item )

if nSamples is not None:
    modFiles = random.sample( modFiles, nSamples )

# ***********************************************************************
# Evaluate
# ***********************************************************************

trendCnts = []
oosErrors = []

for item in modFiles:

    if item.split( '_' )[0] != 'model':
        continue
    
    modFilePath = os.path.join( modDir, item )

    try:
        ecoMfd = dill.load( open( modFilePath, 'rb' ) ).ecoMfd
    except:
        continue
    
    tmp1 = ecoMfd.getOosTrendCnt()
    tmp2 = ecoMfd.getOosError()

    trendCnts.append( tmp1 )    
    oosErrors.append( tmp2 )


print( 'Tend cnt %0.2f +/- %0.2f:' % ( np.mean( trendCnts ), np.std( trendCnts ) ) )
print( 'Oos Error %0.2f +/- %0.2f:' % ( np.mean( oosErrors ), np.std( oosErrors ) ) )
