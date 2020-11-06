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

std_coef   = 1.5
pers_off   = 4

dfFile      = 'data/dfFile_2020.pkl'
initTotVal  = 20000.0
prtFile    = 'portfolios/nTrnDays_360_two_hours_ptc.json'
outPrtFile = 'portfolios/nTrnDays_360_two_hours_ptc_std_coef_%s_pers_off_%s.json' \
    % (str(std_coef), str(pers_off))
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
    
    tmp_df = retDf[ retDf.Date == dates[itr-1] ]
    if tmp_df.shape[0] > 0:
        ret = list(tmp_df.Return)[0]
    else:
        ret = 0.0
        
    tmp_val = ret_mean + std_coef * ret_std
    
    if ret > tmp_val:
        nextDate = dates[itr + pers_off]
        print("Skipping %s" % dates[itr] )
        newWtsHash[ dates[itr] ] = {}
    else:        
        newWtsHash[ dates[itr] ] = prtWtsHash[ dates[itr] ]
        
# ***********************************************************************
# Write the adjusted portfolio
# ***********************************************************************
    
with open( outPrtFile, 'w' ) as fp:
    json.dump( newWtsHash, fp )        
