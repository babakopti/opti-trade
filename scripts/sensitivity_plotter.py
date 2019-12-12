# ***********************************************************************
# Import libraries
# ***********************************************************************

import os
import sys
import dill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    
sys.path.append( os.path.abspath( '../' ) )

from mod.mfdMod import MfdMod

# ***********************************************************************
# Set input parameters
# ***********************************************************************

xType    = 'atnFct'
yType    = 'trend_cnt'
xlog     = False

pltList  = [ { 'model' : '2018-03-10', 'nTrnDays' : '360', 'tol' : '0.05', 'regCoef' : '0.001', 'atnFct' : None } ]

legList  = [ '2018-03-10' ]
             
modDir   = 'models_sensitivity'

figName  = 'atnFct-sensitivity-trend-cnt.png'

title    = 'tol = 0.05; nTrnDays = 360; regCoef = 0.001'

# ***********************************************************************
# Sanity checks + Set some parameters
# ***********************************************************************

assert len( pltList ) == len( legList ), 'Inconsistent size!'

if xType == 'tol':
    xLabel = 'Optimization Tolerance'
elif xType == 'nTrnDays':
    xLabel = 'Num. Training Days'
elif xType == 'regCoef':
    xLabel = 'Regularization Coef.'
elif xType == 'atnFct':
    xLabel = 'Attenuation Factor'    
else:
    assert False, 'Unkown xType!'
    
if yType == 'error':
    yLabel = 'In-Sample Relative Error'
elif yType == 'oos_error':
    yLabel = 'Out-of-Sample Relative Error'
elif yType == 'trend_cnt':
    yLabel = 'Out-of-Sample Trend Match Count'
else:
    assert False, 'Unkown yType!'

# ***********************************************************************
# Some utility functions
# ***********************************************************************

def procFileName( baseName ):

    tmpList = baseName.split( '_' )
    
    if tmpList[0] != 'model' or\
       tmpList[2] != 'nTrnDays' or\
       tmpList[4] != 'tol' or\
       tmpList[6] != 'regCoef' or\
       tmpList[8] != 'atnFct':
        return None

    tmpDict = {}

    for i in range( 0, len( tmpList ), 2 ):
        tmpDict[ tmpList[i] ] = tmpList[i+1]

    return tmpDict

# ***********************************************************************
# plot
# ***********************************************************************

for k in range( len( pltList ) ):
    item  = pltList[k]
    xVals = []
    yVals = []
    for fileName in os.listdir( modDir ):
        tmpList  = os.path.splitext( fileName )
        
        if tmpList[1] != '.dill':
            continue

        baseName = tmpList[0]
        
        tmpDict = procFileName( baseName ) 

        tmpFlag = True
        for tmp in tmpDict:
            
            if tmp == xType:
                continue
            
            if tmpDict[ tmp ] != item[ tmp ]:
                tmpFlag = False
                break
        
        if not tmpFlag:
            continue

        modFile = os.path.join( modDir, fileName )
        mfdMod  = dill.load( open( modFile, 'rb' ) )
        ecoMfd  = mfdMod.ecoMfd

        try:
            tmp = ecoMfd.atnCoefs
        except:
            ecoMfd.atnCoefs = np.ones( shape = ( ecoMfd.nTimes ) )
            
        if yType == 'error':
            yVal = ecoMfd.getError()
        elif yType == 'oos_error':
            yVal = ecoMfd.getOosError()
        elif yType == 'trend_cnt':
            yVal = ecoMfd.getOosTrendCnt()
        else:
            assert False, 'Unkown yType!'

        yVals.append( yVal )
        xVals.append( tmpDict[ xType ] )

    xVals = np.array( xVals, dtype = 'd' )
    yVals = np.array( yVals, dtype = 'd' )

    sortDict = {}
    for j in range( len( xVals ) ):
        sortDict[ yVals[j] ] = xVals[j] 

    xVals = sorted( xVals )
    yVals = sorted( yVals, key = lambda y : sortDict[y] )

    if xlog:
        plt.semilogx( xVals, yVals, 'o-' )
    else:
        plt.plot( xVals, yVals, 'o-' )

plt.title( title )
plt.legend( legList )
plt.xlabel( xLabel )
plt.ylabel( yLabel )
plt.savefig( figName )
plt.show()
