1# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import datetime
import numpy as np
import pandas as pd

from sklearn.metrics import (
    explained_variance_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

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

print(mfdMod.ecoMfd.params)
#mfdMod.ecoMfd.pltResults( rType = 'trn', pType = pType )
mfdMod.ecoMfd.pltResults( rType = 'all', pType = "vel" )

odeObj    = mfdMod.ecoMfd.getSol(mfdMod.ecoMfd.params)
oosOdeObj = mfdMod.ecoMfd.getOosSol()
sol       = odeObj.getSol()
oosSol    = oosOdeObj.getSol()

np.save(open("/Users/babak/Desktop/X_train.npy", "wb"), mfdMod.ecoMfd.actSol.transpose())
np.save(open("/Users/babak/Desktop/X_test.npy", "wb"), mfdMod.ecoMfd.actOosSol.transpose())
# np.save(open("/Users/babak/Desktop/X_train_prd_old.npy", "wb"), sol.transpose())
# np.save(open("/Users/babak/Desktop/X_test_prd_old.npy", "wb"), oosSol.transpose())

for m in range(mfdMod.ecoMfd.nDims):
    
    y_true_train = mfdMod.ecoMfd.actSol[m]
    y_true_test = mfdMod.ecoMfd.actOosSol[m]
    y_pred_train = sol[m]
    y_pred_test = oosSol[m]

    print(
        r2_score(y_true_train, y_pred_train),
        r2_score(y_true_test, y_pred_test)
    )
    
np.save(open("/Users/babak/Desktop/params.npy", "wb"), mfdMod.ecoMfd.params)
