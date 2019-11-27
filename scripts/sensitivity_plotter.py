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
# Set some parameters
# ***********************************************************************

pType    = 'trend_cnt'
xLabel   = 'Optimization Tolerance'

assert pType in [ 'error', 'oos_error', 'trend_cnt' ], 'Unkown pType!'

if pType == 'error':
    yLabel = 'In-Sample Relative Error'
elif pType == 'oos_error':
    yLabel = 'Out-of-Sample Relative Error'
elif pType == 'trend_cnt':
    yLabel = 'Out-of-Sample Trend Match Count'
else:
    assert False, 'Unkown pType!'
 
xVals    = [ 0.01, 0.03, 0.05, 0.1 ]
modFiles = [ 'models_sensitivity/model_2018-03-10_nTrnDays_360_tol_0.01_regCoef_0.001_atnFct_1.0.dill',
             'models_sensitivity/model_2018-03-10_nTrnDays_360_tol_0.03_regCoef_0.001_atnFct_1.0.dill',
             'models_sensitivity/model_2018-03-10_nTrnDays_360_tol_0.05_regCoef_0.001_atnFct_1.0.dill',
             'models_sensitivity/model_2018-03-10_nTrnDays_360_tol_0.1_regCoef_0.001_atnFct_1.0.dill'  ]

assert len( xVals ) == len( modFiles ), 'Inconsistent sizes!'

# ***********************************************************************
# plot
# ***********************************************************************

yVals = []

for modFile in modFiles:

    
    mfdMod = dill.load( open( modFile, 'rb' ) )

    ecoMfd = mfdMod.ecoMfd
    
    if pType == 'error':
        yVal = ecoMfd.getError()
    elif pType == 'oos_error':
        yVal = ecoMfd.getOosError()
    elif pType == 'trend_cnt':
        yVal = ecoMfd.getOosTrendCnt()
    else:
        assert False, 'Unkown pType!'
        
    yVals.append( yVal )

plt.plot( xVals, yVals, 'o-' )
plt.xlabel( xLabel )
plt.ylabel( yLabel )
plt.show()
