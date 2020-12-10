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
from dat.assets import ETF_HASH

# ***********************************************************************
# Main input params
# ***********************************************************************

STD_COEF = 1.8
PERS_OFF = 11

DF_FILE = 'data/dfFile_2020.pkl'
PRT_FILE = 'portfolios/nTrnDays_360_two_hours_ptc.json'
OUT_PRT_FILE = 'portfolios/nTrnDays_360_two_hours_ptc_lsp_%s_%s.json' \
    % (str(STD_COEF), str(PERS_OFF))

# ***********************************************************************
# Read original portfolio and get symbols and adjust
# ***********************************************************************

prtWtsHash = json.load( open( PRT_FILE, 'r' ) )

# ***********************************************************************
# Adjust
# ***********************************************************************

def adjustLsp( std_coef, pers_off, prtWtsHash, dfFile ):

    retDf = utl.calcBacktestReturns( prtWtsHash = prtWtsHash,
                                     dfFile     = dfFile,
                                     initTotVal = 20000,
                                     shortFlag  = False,
                                     invHash    = ETF_HASH,
                                     minDate    = None,
                                     maxDate    = None   )

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
        
        tmp_val = ret_mean - std_coef * ret_std
    
        if ret < tmp_val:
            offset = min(len(dates)-itr-1, pers_off)
            nextDate = dates[itr + offset]
            print("Skipping %s" % dates[itr] )
            newWtsHash[ dates[itr] ] = {}
        else:        
            newWtsHash[ dates[itr] ] = prtWtsHash[ dates[itr] ]

    return newWtsHash

# ***********************************************************************
# Adjust
# ***********************************************************************

def getLspPerf( std_coef, pers_off, prtWtsHash, dfFile ):

    newWtsHash = adjustLsp( std_coef, pers_off, prtWtsHash, dfFile )

    retDf = utl.calcBacktestReturns( prtWtsHash = newWtsHash,
                                     dfFile     = dfFile,
                                     initTotVal = 20000,
                                     shortFlag  = False,
                                     invHash    = ETF_HASH,
                                     minDate    = None,
                                     maxDate    = None   )
    
    return ( retDf.Return.mean() / retDf.Return.std() )

# ***********************************************************************
# Write the adjusted portfolio
# ***********************************************************************

newWtsHash = adjustLsp( STD_COEF, PERS_OFF, prtWtsHash, DF_FILE )

with open( OUT_PRT_FILE, 'w' ) as fp:
    json.dump( newWtsHash, fp )        

print(getLspPerf(STD_COEF, PERS_OFF, prtWtsHash, DF_FILE ))
    
