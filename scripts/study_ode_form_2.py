import os
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

nDims = 2
bcTime = 0.0
endTime = 2.6
nTimes = 1000
bcVec = np.array([ 1.0, 1.0, 0.0, 0.0 ])
Gamma = np.zeros( shape = ( nDims, nDims, nDims ), dtype = 'd' )
beta = np.zeros( shape = ( nDims ), dtype = 'd' )
zeta = np.zeros( shape = ( nDims ), dtype = 'd' )
eta = np.zeros( shape = ( nDims ), dtype = 'd' )

Gamma[0][0][0] = 1.0
Gamma[1][1][1] = 1.0

Gamma[0][0][1] = -2.0
Gamma[1][0][1] = 1.2

Gamma[0][1][1] = 0.
Gamma[1][0][0] = 0

Gamma[0][1][0] = Gamma[0][0][1]
Gamma[1][1][0] = Gamma[1][0][1]

beta[0] = 2.4
beta[1] = 0

zeta[0] = 1.5
zeta[1] = 0

eta[0] = 0.0
eta[1] = 0

def fun( t, X ):

    vals = np.zeros( shape = ( 2 * nDims ), dtype = 'd' )
    y = np.zeros( shape = ( nDims ), dtype = 'd' )

    for m in range( nDims ):
        y[m] = X[m + nDims]

    gammaTerm = np.tensordot( Gamma,
                              np.tensordot( y, y, axes = 0 ),
                              ( (1,2), (0,1) ) )
    
    for m in range( nDims ):
        vals[m] = y[m]
        vals[m + nDims] = -gammaTerm[m] - beta[m] * (X[m] - zeta[m] - eta[m] * t)
        
    return vals

timeSpan = ( bcTime, endTime )        
timeEval = np.linspace( bcTime, endTime, nTimes )

res = solve_ivp( fun      = fun, 
                 y0       = bcVec, 
                 t_span   = timeSpan,
                 t_eval   = timeEval,
                 method   = 'LSODA', 
                 rtol     = 1.0e-6            )

print( 'Success:', res.success )

plt.plot(res.y[0])
#plt.plot(res.y[1])
plt.show()
