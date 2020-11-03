# ***********************************************************************
# Import libraries
# ***********************************************************************

import os
import sys
import ast
import json
import pickle
import numpy as np
import pandas as pd

sys.path.append( '..' )

import utl.utils as utl

# ***********************************************************************
# Main input params
# ***********************************************************************

std_coef   = 1.0
days_off   = 5

dfFile      = 'data/dfFile_crypto.pkl'
initTotVal  = 20000.0
prtFile    = 'portfolios/crypto_9PM_ptc_no_short.json'
outPrtFile = 'portfolios/crypto_9PM_ptc_std_coef_%s_days_off_%s.json' \
    % (str(std_coef), str(days_off))
minDate    = None
maxDate    = None

# ***********************************************************************
# Read original portfolio and get symbols
# ***********************************************************************

prtWtsHash = json.load( open( prtFile, 'r' ) )

# ***********************************************************************
# Adjust
# ***********************************************************************

retDf = utl.calcBacktestReturns( prtWtsHash = prtWtsHash,
                                 dfFile     = dfFile,
                                 initTotVal = initTotVal,
                                 shortFlag  = True,
                                 invHash    = None,
                                 minDate    = minDate,
                                 maxDate    = maxDate   )

ret_mean = retDf.Return.mean()
ret_std  = retDf.Return.std()

newWtsHash = {}

dates = sorted( prtWtsHash.keys() )

skipFlag = False
nextDate = None

for itr in range(1, len(dates)):

    if nextDate is not None and dates[itr] < nextDate:
        print("Skipping %s" % dates[itr] )
        newWtsHash[ dates[itr] ] = {}
        continue
    
    ret = list(retDf[ retDf.Date == dates[itr-1] ].Return)[0]
    tmp_val = ret_mean + std_coef * ret_std
    
    if ret > tmp_val:
        nextDate = dates[itr + days_off]
        print("Skipping %s" % dates[itr] )
        newWtsHash[ dates[itr] ] = {}
    else:        
        newWtsHash[ dates[itr] ] = prtWtsHash[ dates[itr] ]
        
# ***********************************************************************
# Write the adjusted portfolio
# ***********************************************************************
    
with open( outPrtFile, 'w' ) as fp:
    json.dump( newWtsHash, fp )        
