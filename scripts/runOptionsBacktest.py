# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import datetime
import logging
import json
import numpy as np
import pandas as pd

from unittest import mock

sys.path.append( os.path.abspath( '..' ) )

import utl.utils as utl

from prt.prt import MfdOptionsPrt 

# ***********************************************************************
# Main input params
# ***********************************************************************

prtFile     = 'portfolios/options.json'
bkBegDate   = pd.to_datetime( '2020-10-10 09:30:00' )
bkEndDate   = pd.to_datetime( '2020-11-04 15:30:00' )

# ***********************************************************************
# Set some parameters and read data
# ***********************************************************************

MIN_PROB  = 0.51
MAX_COST  = 50.0
TRADE_FEE = 2 * 0.75

MIN_HORIZON = 1
MAX_HORIZON = 10

OPTION_CHAIN_FILE = 'options_chain_data/option_chain_Aug_Nov_2020.pkl'
    
MAX_TRIES = 100
NUM_DAILY_TRADES = 1

logger = utl.getLogger( None, 1 )

optDf = pd.read_pickle( OPTION_CHAIN_FILE )

dates = sorted( set(optDf.DataDate) )

# ***********************************************************************
# Run the backtest
# ***********************************************************************

MfdOptionsPrt.getProb = lambda self, x: x[ 'prob' ]

for curDate in dates:
    logger.info( 'Processing %s', str( curDate ) )

    minDate = curDate + datetime.timedelta( days = MIN_HORIZON )
    minDate = curDate + datetime.timedelta( days = MAX_HORIZON )

    with patch.object(MfdOptionsPrt, 'setMod', return_value=None):
        with patch.object(MfdOptionsPrt, 'setPrdDf', return_value=None):
            prtObj = MfdOptionsPrt(
                modFile      = None,
                assetHash    = None,
                curDate      = currDate,
                minDate      = minDate,
                maxDate      = maxDate,
                minProb      = MIN_PROB,,
                rfiDaily     = 0.0,
                tradeFee     = TRADE_FEE,
                nDayTimes    = 1140,
                logFileName  = None,                    
                verbose      = 1          )

    tmpDf = optDf[ optDf.DataDate == curDate ]
    tmpDf = tmpDf.rename(
        columns = { 'Prob' : 'prob' }
    )
    tmpDf = tmpDf[ [ 'optionSymbol',
                     'assetSymbol',
                     'type',
                     'strike',
                     'expiration',
                     'unitPrice',
                     'prob'     ] ]
    options = [
        dict( item ) for ind, item in tmpDf.iterrows()
    ]
    
    callDf = prtObj.selVosPairs( 
        options,
        'call',
        MAX_COST,
        NUM_DAILY_TRADES,
        MAX_TRIES,
    )

    putDf = prtObj.selVosPairs( 
        options,
        'put',
        MAX_COST,
        NUM_DAILY_TRADES,
        MAX_TRIES,
    )
    
    df = pd.concat( [ callDf, putDf ] )
    df = df.sort_values( 'maxReturn', ascending = False )
    df = df.head( MAX_DAILY_TRADE )

    selList = [ dict( item ) for ind, item in df.iterrows() ]

    
