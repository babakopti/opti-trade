# ***********************************************************************
# Import libraries
# ***********************************************************************

import os
import sys
import dill
import pickle
import time
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

sys.path.append( os.path.abspath( '../' ) )

from mod.mfdMod import MfdMod

from ode.odeGeo import OdeGeoConst
from ode.odeGeo import OdeAdjConst

# ***********************************************************************
# Some definitions
# ***********************************************************************

mfdMod = dill.load( open( 'model_2015-01-01_2015-12-31.dill', 'rb' ) )
ecoMfd = mfdMod.ecoMfd
Gamma  = ecoMfd.getGammaArray( ecoMfd.GammaVec )
bcTime = list( ecoMfd.trnDf.time )[-1]
nSols  = 1

# ***********************************************************************
# Parameters
# ***********************************************************************

tol      = 1.0e-3
nMaxItrs = 1000

# ***********************************************************************
# Solve and record time 
# ***********************************************************************

t0     = time.time()

for i in range( nSols ):
    odeObj   = OdeGeoConst(  Gamma    = Gamma,
                             bcVec    = ecoMfd.bcSol,
                             bcTime   = bcTime,
                             timeInc  = 1.0,
                             nSteps   = ecoMfd.nSteps,
                             intgType = 'LSODA', 
                             tol      = tol, 
                             nMaxItrs = nMaxItrs         )
    sFlag = odeObj.solve()

    sol   = odeObj.getSol()

print( 'OdeGeoConst took %0.2f seconds!' % 
       round( time.time() - t0, 2 ) )

bcVec  = np.zeros( shape = ( ecoMfd.nDims ), dtype = 'd' )
t0     = time.time()

for i in range( nSols ):
    adjOdeObj = OdeAdjConst( Gamma     = Gamma,
                             bcVec     = bcVec,
                             bcTime    = 0.0,
                             timeInc   = 1.0,
                             nSteps    = ecoMfd.nSteps,
                             intgType  = 'RK45',
                             actSol    = ecoMfd.actSol,
                             adjSol    = sol,
                             tol       = tol,
                             nMaxItrs  = 1000,
                             verbose   = ecoMfd.verbose       )

    sFlag  = adjOdeObj.solve()

    adjSol = adjOdeObj.getSol()

print( 'OdeAjdConst took %0.2f seconds!' % 
       round( time.time() - t0, 2 ) )

