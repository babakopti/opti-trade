import os
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

sys.path.append( '..' )

from ode.odeGeo import OdeGeoConst2

nDims = 2
bcTime = 0.0
endTime = 2.6
nTimes = 1001
bcVec = np.array([ 0.5, 0.7, 0.01, 0.01 ])
Gamma = np.zeros( shape = ( nDims, nDims, nDims ), dtype = 'd' )
beta = np.zeros( shape = ( nDims ), dtype = 'd' )
actAvgSol = np.ones( shape = ( nDims, nTimes ), dtype = 'd' )

Gamma[0][0][0] = 0
Gamma[1][1][1] = 0.

Gamma[0][0][1] = -2.0
Gamma[1][0][1] = 1.2

Gamma[0][1][1] = 0.
Gamma[1][0][0] = 0

Gamma[0][1][0] = Gamma[0][0][1]
Gamma[1][1][0] = Gamma[1][0][1]

beta[0] = -0.00001
beta[1] = 0

odeObj = OdeGeoConst2( Gamma     = Gamma,
                       beta      = beta,
                       bcVec     = bcVec,
                       bcTime    = bcTime,
                       timeInc   = 1.0,
                       nSteps    = nTimes-1,
                       intgType  = 'LSODA',
                       actAvgSol = actAvgSol,
                       verbose   = 1           )

sFlag = odeObj.solve()

sol = odeObj.getSol()

plt.plot(sol[0])
plt.plot(sol[1])
plt.show()
