# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import dill
import time
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

sys.path.append( os.path.abspath( '../' ) )

import utl.utils as utl
from mod.mfdMod import MfdMod

# ***********************************************************************
# Get model and process
# ***********************************************************************

modFile  = 'models/model.dill'

t0       = time.time()
print( 'Reading model file...' )

mfdMod   = dill.load( open( modFile, 'rb' ) )

t        = round( time.time()-t0, 1 )
print( 'Loading model file took', t, 'seconds!' )

t0       = time.time()
print( 'Getting Gamma...' )

ecoMfd   = mfdMod.ecoMfd

ecoMfd.logFileName = None
ecoMfd.logger = utl.getLogger( None, 1 )

nDims    = ecoMfd.nDims
nTimes   = ecoMfd.nTimes
nSteps   = ecoMfd.nSteps
velNames = ecoMfd.velNames
varNames = ecoMfd.varNames
actSol   = ecoMfd.actSol
odeObj   = ecoMfd.getSol( ecoMfd.GammaVec )
sol      = odeObj.getSol()
Gamma    = ecoMfd.getGammaArray( ecoMfd.GammaVec )

nTimes   = int( nTimes / 10 )
res      = np.zeros( shape = ( nDims, nTimes ), dtype = 'd' )
times    = np.linspace( 0, nSteps, nTimes )
tmpVec   = np.zeros( shape = ( nDims ), dtype = 'd' )
tmpActVec= np.zeros( shape = ( nDims ), dtype = 'd' )

t        = round( time.time()-t0, 1 )
print( 'Getting Gamma took', t, 'seconds!' )

# ***********************************************************************
# Calculate residuals
# ***********************************************************************

t0 = time.time()

print( 'Calculating residuals..' )

for tsId in range( nTimes - 1 ):
    
    for m in range( nDims ):

        res[m][tsId] = actSol[m][tsId + 1] - actSol[m][tsId] - ( sol[m][tsId + 1] - sol[m][tsId] )

        for b in range( nDims ):
            tmpVec[b]    = sol[b][tsId]
            tmpActVec[b] = actSol[b][tsId]

        for a in range( nDims ):
            res[m][tsId] += np.dot( Gamma[m][a][:] * actSol[a][tsId], tmpActVec ) -\
                    np.dot( Gamma[m][a][:] * sol[a][tsId], tmpVec )

t = round( time.time()-t0, 1 )

print( 'Calculating residuals took', t, 'seconds!' )

# ***********************************************************************
# Plot f(s) = residual[m] / y[m]
# ***********************************************************************

y = np.empty( shape = ( nDims, nTimes ), dtype = 'd' )

for tsId in range( nTimes ):
    for m in range( nDims ):
        tmp = actSol[m][tsId]
        if tmp != 0:
            tmp = 1.0 / tmp
        y[m][tsId] = res[m][tsId] * tmp

y_avg = np.zeros( shape = ( nTimes ), dtype = 'd' )
for tsId in range( nTimes ):
    y_avg[tsId] = 0.0
    for m in range( nDims ):
        y_avg[tsId] += y[m][tsId] / nDims

resDf = pd.DataFrame( { 'average' : y_avg } )

for m in range( nDims ):
    resDf[ velNames[m] ] = y[m]

resDf.to_pickle( 'f_of_s_model_2019-03-18.pkl', protocol = 4 )

plt.plot( times[-1000:], y_avg[-1000:] )
    
plt.xlabel( 'Time' )
plt.ylabel( 'f(s)' )
plt.show()

sys.exit()

# ***********************************************************************
# Try to fit the source term (ODE residual)
# ***********************************************************************

linReg = LinearRegression( fit_intercept = True )
resPrd = np.zeros( shape = ( nDims, nTimes ), dtype = 'd' )
coefs  = np.zeros( shape = ( nDims, 2 ), dtype = 'd' )

for m in range( nDims ):

    X = np.reshape( actSol[m][:nTimes], ( nTimes, 1 ) )

    linReg.fit( X, res[m] )

    print( 'R2 =', linReg.score( X, res[m] ) )

    assert len( linReg.coef_ ) == 1, 'Incorrect size!'

    coefs[m][0] = linReg.coef_[0]

    coefs[m][1] = linReg.intercept_

    yPrd = linReg.predict( X )

    for a in range( nTimes ):
        resPrd[m][tsId] = yPrd[tsId]

# ***********************************************************************
# Plot
# ***********************************************************************

for m in range( nDims ): 

    plt.plot( times, res[m],
              times, resPrd[m] )

    plt.xlabel( 'Time' )
    plt.ylabel( 'ODE res. ' + velNames[m] )
    plt.show()
