# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import datetime
import logging
import dill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import trapz

sys.path.append( os.path.abspath( '../' ) )

from mod.mfdMod import MfdMod

# ***********************************************************************
# Set some parameters and read data
# ***********************************************************************

modFlag     = True

nBoosts     = 5
dfFile      = 'data/dfFile_2016plus.pkl'

minTrnDate  = pd.to_datetime( '2017-02-01 09:00:00' )
maxTrnDate  = pd.to_datetime( '2018-01-31 09:00:00' )
maxOosDate  = pd.to_datetime( '2018-02-10 23:59:00' )

indices     = [ 'INDU', 'NDX', 'SPX', 'RUT', 'OEX',  
                'MID',  'SOX', 'RUI', 'RUA', 'TRAN',
                'HGX',  'TYX', 'XAU'               ] 
futures     = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]
ETFs        = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH', 
                'SMH', 'XLE', 'XLF', 'XLU', 'EWJ'          ]
velNames    = ETFs + indices + futures

# ***********************************************************************
# Utility functions
# ***********************************************************************

def getPointError( sol, actSol, tsId ): 

    nDims = sol.shape[0]

    fct = 0.0
    for varId in range( nDims ):
        fct += actSol[varId][tsId]**2 

    if  fct > 0:
        fct = 1.0 / fct

    val = 0.0
    for varId in range( nDims ):
        val += ( sol[varId][tsId] - actSol[varId][tsId] )**2 

    return min(fct * val, 0.999)

def getTotError( sol, actSol, atnCoefs ): 

    nTimes = sol.shape[1]

    if atnCoefs is None:
        atnCoefs = np.ones( shape = ( nTimes ), dtype = 'd' )

    atnCoefs = atnCoefs / np.sum( atnCoefs )
    
    val = 0.0
    for tsId in range( nTimes ):
        err  = getPointError( sol, actSol, tsId )
        val += atnCoefs[tsId] * err
        
    return val
    
# ***********************************************************************
# Create model builder object
# ***********************************************************************

if modFlag: 
    mfdMod = MfdMod(    dfFile       = dfFile,
                        minTrnDate   = minTrnDate,
                        maxTrnDate   = maxTrnDate,
                        maxOosDate   = maxOosDate,
                        optType      = 'SLSQP',
                        velNames     = velNames,
                        maxOptItrs   = 100,
                        optGTol      = 1.0e-2,
                        optFTol      = 1.0e-2,
                        factor       = 4.0e-5,
                        regCoef      = 1.0e-3,
                        smoothCount  = None,
                        logFileName  = None,
                        verbose      = 1        )

    validFlag = mfdMod.build()
else:
    mfdMod = dill.load( open( 'model_boost_0.dill', 'rb' ) )

nTimes    = mfdMod.ecoMfd.nTimes
nOosTimes = mfdMod.ecoMfd.nOosTimes
nDims     = mfdMod.ecoMfd.nDims
actSol    = mfdMod.ecoMfd.actSol
actOosSol = mfdMod.ecoMfd.actOosSol

# ***********************************************************************
# Build models with boosting
# ***********************************************************************

atnCoefs  = np.ones( shape = ( nTimes ), dtype = 'd' )
finSol    = np.zeros( shape = ( nDims, nTimes ), dtype = 'd' )
finOosSol = np.zeros( shape = ( nDims, nOosTimes ), dtype = 'd' )
totAlpha  = 0.0

X = []
inErrs = []
outErrs = []
alphas = []

for boostId in range( nBoosts ):

    # Normalize weights

    fct = np.sum( atnCoefs )

    if fct > 0:
        fct = 1.0 / fct
    else:
        assert False, 'Incorrect weights!'

    atnCoefs = fct * atnCoefs

    # Build the new model
    if modFlag:
        mfdMod = MfdMod(dfFile       = dfFile,
                        minTrnDate   = minTrnDate,
                        maxTrnDate   = maxTrnDate,
                        maxOosDate   = maxOosDate,
                        optType      = 'SLSQP',
                        velNames     = velNames,
                        maxOptItrs   = 100,
                        optGTol      = 1.0e-2,
                        optFTol      = 1.0e-2,
                        factor       = 4.0e-5,
                        regCoef      = 1.0e-3,
                        atnCoefs     = atnCoefs,
                        smoothCount  = None,
                        logFileName  = None,
                        verbose      = 0        )        
        validFlag = mfdMod.build()
        
        mfdMod.save( 'model_boost_%d.dill' % boostId )
    else:
        mfdMod = dill.load( open( 'model_boost_%d.dill' % boostId, 'rb' ) )
        validFlag = True

    sol    = mfdMod.ecoMfd.getSol( mfdMod.ecoMfd.GammaVec ).getSol()
    oosSol = mfdMod.ecoMfd.getOosSol().getSol()

    err = getTotError( sol, actSol, atnCoefs )

    assert err < 1.0, 'Error should be < 1.0!'
    
    # Calculate alpha
    alpha  = np.log( ( 1.0 - err ) / err )

    alphas.append( alpha )
    
    # Get the weights for next iteration
    
    for tsId in range( nTimes ):
        loss = getPointError( sol, actSol, tsId )
        atnCoefs[tsId] = atnCoefs[tsId] * np.exp( alpha * loss )

    logging.info( 'Boost %d Success: %s, error: %0.3f, alpha: %0.3f',
                  boostId + 1, validFlag, err, alpha )

    # Get solutions after boosting
    
    finSol = finSol + alpha * sol

    finOosSol = finOosSol + alpha * oosSol
            
    totAlpha += alpha

    inErr = getTotError( finSol / totAlpha, actSol, None )
    
    logging.info( 'In-sample error after %d boosting iteations: %0.3f',
                  boostId + 1, inErr )

    outErr = getTotError( finOosSol / totAlpha, actOosSol, None )

    logging.info( 'Out-of-sample error after %d boosting iterations: %0.3f',
                  boostId + 1,
                  outErr )

    X.append( boostId + 1 )
    inErrs.append( inErr )
    outErrs.append( outErr )
    
logging.info( 'Done with %d iteartions of boosting!', nBoosts )

plt.plot( X, inErrs, 'b-o' )
plt.xlabel( 'Boosting Iterations' )
plt.ylabel( 'In-sample Error' )
plt.show()

plt.plot( X, outErrs, 'r-s' )
plt.xlabel( 'Boosting Iterations' )
plt.ylabel( 'Out-of-sample Error' )
plt.show()

print( alphas )
print( totAlpha, np.sum( alphas ) )

print('inErrs: ', inErrs )
print('outErrs: ', outErrs )
