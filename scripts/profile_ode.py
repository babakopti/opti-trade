# ***********************************************************************
# Import libraries
# ***********************************************************************

import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
import scipy as sp

sys.path.append( os.path.abspath( '../' ) )

from ode.odeGeo import OdeGeoConst

# ***********************************************************************
# Some definitions
# ***********************************************************************

Gamma  = pickle.load( open( 'gamma.pkl', 'rb' ) )
bcTime = 66545.0
bcSol  = np.array([6.56000000e-05, 6.13789778e-05, 7.35751295e-05])
nSteps = 66545
nSols  = 20

# ***********************************************************************
# Parameters
# ***********************************************************************

tol      = 1.0e-6
nMaxItrs = 1000

# ***********************************************************************
# Solve and record time
# ***********************************************************************

t0     = time.time()

for i in range( nSols ):
    odeObj   = OdeGeoConst(  Gamma    = Gamma,
                             bcVec    = bcSol,
                             bcTime   = bcTime,
                             timeInc  = 1.0,
                             nSteps   = nSteps,
                             intgType = 'vode', 
                             tol      = tol, 
                             nMaxItrs = nMaxItrs         )
    sFlag = odeObj.solve()

print( sFlag )

print( 'Solutions took %0.2f seconds!' % 
       round( time.time() - t0, 2 ) )
