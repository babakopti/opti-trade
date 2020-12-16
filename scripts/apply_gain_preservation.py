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

STD_COEF = 1.4
PERS_OFF = 4
NUM_PERS = 30

SHORT_FLAG = True

DF_FILE = 'data/dfFile_crypto.pkl'
PRT_FILE = 'portfolios/crypto_9PM_no_zcash_ptc_no_short.json'
OUT_PRT_FILE = 'portfolios/crypto_9PM_no_zcash_ptc_no_short_gnp_%s_%s_%s.json' \
    % (str(STD_COEF), str(PERS_OFF), str(NUM_PERS))

# ***********************************************************************
# Read original portfolio and get symbols and adjust
# ***********************************************************************

prtWtsHash = json.load( open( PRT_FILE, 'r' ) )

# ***********************************************************************
# Adjust
# ***********************************************************************

def adjustGnp( std_coef, pers_off, num_pers, prtWtsHash, dfFile ):

    retDf = utl.calcBacktestReturns( prtWtsHash = prtWtsHash,
                                     dfFile     = dfFile,
                                     initTotVal = 20000,
                                     shortFlag  = SHORT_FLAG,
                                     invHash    = ETF_HASH,
                                     minDate    = None,
                                     maxDate    = None   )

    newWtsHash = {}

    dates = sorted( prtWtsHash.keys() )

    skipFlag = False
    nextDate = None

    for itr in range(num_pers + 1, len(dates)):

        ret_mean = retDf.head(itr-1).tail(num_pers).Return.mean()
        ret_std  = retDf.head(itr-1).tail(num_pers).Return.std()
        
        if nextDate is not None and dates[itr] < nextDate:
            newWtsHash[ dates[itr] ] = {}
            continue
    
        tmp_df = retDf[ retDf.Date == dates[itr-1] ]
        if tmp_df.shape[0] > 0:
            ret = list(tmp_df.Return)[0]
        else:
            ret = 0.0
        
        tmp_val = ret_mean + std_coef * ret_std
    
        if ret > tmp_val:
            offset = min(len(dates)-itr-1, pers_off)
            nextDate = dates[itr + offset]
            newWtsHash[ dates[itr] ] = {}
            retDf['Return'] = retDf.apply(
                lambda x: 0.0 if (x.Date >= pd.to_datetime(dates[itr]) and \
                                  x.Date < pd.to_datetime(nextDate)) \
                else x.Return, axis=1
            )
        else:        
            newWtsHash[ dates[itr] ] = prtWtsHash[ dates[itr] ]

    return newWtsHash

# ***********************************************************************
# Adjust
# ***********************************************************************

def getGnpPerf( std_coef, pers_off, num_pers, prtWtsHash, dfFile ):

    newWtsHash = adjustGnp( std_coef, pers_off, num_pers, prtWtsHash, dfFile )

    retDf = utl.calcBacktestReturns( prtWtsHash = newWtsHash,
                                     dfFile     = dfFile,
                                     initTotVal = 20000,
                                     shortFlag  = SHORT_FLAG,
                                     invHash    = ETF_HASH,
                                     minDate    = None,
                                     maxDate    = None   )
    
    return ( retDf.Return.mean() / retDf.Return.std() )

# ***********************************************************************
# Write the adjusted portfolio
# ***********************************************************************

if __name__ == '__main__':
    newWtsHash = adjustGnp( STD_COEF, PERS_OFF, NUM_PERS, prtWtsHash, DF_FILE )

    with open( OUT_PRT_FILE, 'w' ) as fp:
        json.dump( newWtsHash, fp )        

    print(getGnpPerf(STD_COEF, PERS_OFF, NUM_PERS, prtWtsHash, DF_FILE ))
    
