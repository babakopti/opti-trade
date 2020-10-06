import os
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

bcTime = 0.0
endTime = 10.0
nTimes = 1000
bcVec = np.array([ 1.0, 1.0 ])
Gamma = np.zeros( shape = ( 2, 2, 2 ), dtype = 'd' )

Gamma[0][0][0] = 0.5
Gamma[1][1][1] = 1.0

Gamma[0][0][1] = -1.0
Gamma[1][0][1] = 0.2

Gamma[0][1][1] = 0.5
Gamma[1][0][0] = 0.1

Gamma[0][1][0] = Gamma[0][0][1]
Gamma[1][1][0] = Gamma[1][0][1]

def fun( t, y ):
    srcTerm = np.zeros( shape = ( 2 ), dtype = 'd' )
#    if y[0] >= 1.5:
#        srcTerm[0] = -10.0 * ( y[0] - 2.5 )

    return -np.tensordot( Gamma,
                          np.tensordot( y, y, axes = 0 ),
                          ( (1,2), (0,1) ) ) + srcTerm

def jac( t, y ):
    return -2.0 * np.tensordot( Gamma, y, axes = ( (2), (0) ) )

def solve_geodesic( bcTime,
                    endTime,
                    nTimes,
                    bcVec,
                    Gamma  ):

    timeSpan = ( bcTime, endTime )        
    timeEval = np.linspace( bcTime, endTime, nTimes )

    res = solve_ivp( fun      = fun, 
                     #jac      = jac,
                     y0       = bcVec, 
                     t_span   = timeSpan,
                     t_eval   = timeEval,
                     method   = 'LSODA', 
                     rtol     = 1.0e-6     )

    print( 'Success:', res.success )
    y = res.y #np.flip( res.y, 1 )

    plt.plot(y[0], 'b' )
    plt.plot(y[1], 'r' )
    plt.legend([str(Gamma[0]), str(Gamma[1])])
    plt.show()

for itr in range( 20 ):

    Gamma[0] = 0.5 * ( 2 * np.random.rand(2, 2) - 1 )
    Gamma[1] = 0.5 * ( 2 * np.random.rand(2, 2) - 1 )

    Gamma[0][1][0] = Gamma[0][0][1]
    Gamma[1][1][0] = Gamma[1][0][1]

    solve_geodesic( bcTime  = bcTime,
                    endTime = endTime,
                    nTimes  = nTimes,
                    bcVec   = bcVec,
                    Gamma   = Gamma    )    
