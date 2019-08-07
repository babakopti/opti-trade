# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os

sys.path.append( os.path.abspath( '../' ) )

from mod.mfdMod import MfdMod

import numpy as np
import pandas as pd

# ***********************************************************************
# Set some parameters and read data
# ***********************************************************************

cumulFlag   = True
dfFile      = 'data/vars.csv'
minTrnDate  = '2000-01-01'
maxTrnDate  = '2016-12-31'
maxOosDate  = '2019-05-01'

varNames    = None
modFileName = 'model_' + minTrnDate + '_' + maxTrnDate + '.dill'

# if cumulFlag:
#     varNames    = [ 'DJIA_Cumul', 
#                     'SP500_CF_Cumul', 
#                     'NASDAQ100_EMINI_CF_Cumul', 
#                     'AA_BOND_Cumul' ]
    
#     modFileName = 'test.dill'
# else:
#     varNames    = [ 'DJIA', 
#                     'SP500_CF', 
#                     'NASDAQ100_EMINI_CF', 
#                     'AA_BOND' ]

#     modFileName = 'test_non_cumul.dill'

# ***********************************************************************
# Build model
# ***********************************************************************

mfdMod = MfdMod(    dfFile       = dfFile,
                    minTrnDate   = minTrnDate,
                    maxTrnDate   = maxTrnDate,
                    maxOosDate   = maxOosDate,
                    varNames     = varNames,
                    maxOptItrs   = 2000,
                    optGTol      = 1.0e-2,
                    optFTol      = 1.0e-2,
                    regCoef      = 1.0e-4,
                    minMerit     = 0.65,
                    maxBias      = 0.10,
                    varFiltFlag  = True,
                    validFlag    = True,
                    smoothDays   = 30,
                    cumulFlag    = cumulFlag,
                    verbose      = 1          )

validFlag = mfdMod.build()

print( 'Success :', validFlag )

mfdMod.save( modFileName )

#mfdMod.ecoMfd.pltResults( rType = 'oos', pType = 'vel' )
#mfdMod.ecoMfd.pltResults( rType = 'oos', pType = 'var' )
mfdMod.ecoMfd.pltResults( rType = 'all', pType = 'vel' )
#mfdMod.ecoMfd.pltResults( rType = 'all', pType = 'var' )

#print( mfdMod.ecoMfd.getMeanErrs( rType = 'oos', vType = 'vel' ) )
#print( mfdMod.ecoMfd.getMeanErrs( rType = 'oos', vType = 'var' ) )

