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
import matplotlib.pyplot as plt

from unittest.mock import patch

sys.path.append( os.path.abspath( '..' ) )

import utl.utils as utl

from prt.prt import MfdOptionsPrt 

# ***********************************************************************
# Main input params
# ***********************************************************************

OUT_CSV_FILE = 'data/options_trade_blances.csv'
OPTION_CHAIN_FILE = 'options_chain_data/option_chain_Aug_Nov_2020.pkl'
ACT_FILE = 'data/dfFile_2020-11-10 15:00:06.pkl'

MIN_PROB  = 0.49
MAX_PRICE_C  = 500.0
MAX_PRICE_A  = 5000.0
TRADE_FEE = 2 * 0.75
INIT_CASH = 5000

MIN_HORIZON = 1
MAX_HORIZON = 10

MAX_TRIES = 1000
NUM_DAILY_TRADES = 1

# ***********************************************************************
# Read scored options chains data 
# ***********************************************************************

optDf = pd.read_pickle( OPTION_CHAIN_FILE )
actDf = pd.read_pickle( ACT_FILE )

optDf[ 'DataDate' ] = optDf.DataDate.astype( 'datetime64[ns]' )
optDf[ 'expiration' ] = optDf.expiration.astype( 'datetime64[ns]' )

# ***********************************************************************
# Read actuals and set a hash table
# ***********************************************************************

actDf[ 'Date' ] = actDf.Date.astype( 'datetime64[ns]' )
actDf[ 'Date' ] = actDf.Date.apply( lambda x: x.strftime( '%Y-%m-%d' ) )
actDf[ 'Date' ] = actDf.Date.astype( 'datetime64[ns]' )

actDf = actDf.groupby( 'Date', as_index = False ).mean()

ACT_HASH = {}
for symbol in set( optDf.assetSymbol ):
    ACT_HASH[ symbol ] = dict( zip( actDf[ 'Date' ], actDf[ symbol ] ) )

# ***********************************************************************
# Set some parameters
# ***********************************************************************

dates = sorted( set( optDf.DataDate ) )

logger = utl.getLogger( None, 1 )

# ***********************************************************************
# Initialize the holding hash
# ***********************************************************************

holdHash = {}
balHash = {}

for date in dates:
    
    holdHash[ date ] = {
        'cash': None,
        'options': [],
    }

# ***********************************************************************
# Some utility functions
# ***********************************************************************

def getOptions( curDate, optDf ):
    
    tmpDf = optDf[ optDf.DataDate == curDate ]
    tmpDf = tmpDf.rename(
        columns = { 'Prob' : 'prob' }
    )
    tmpDf = tmpDf[ [ 'optionSymbol',
                     'assetSymbol',
                     'type',
                     'contractCnt',
                     'strike',
                     'expiration',
                     'unitPrice',
                     'prob'     ] ]
    
    options = [
        dict( item ) for ind, item in tmpDf.iterrows()
    ]

    return options

def selTradeOptions( curDate, holdHash, optDf ):
    
    minDate = curDate + datetime.timedelta( days = MIN_HORIZON )
    maxDate = curDate + datetime.timedelta( days = MAX_HORIZON )

    assetHash = {}

    for symbol in ACT_HASH:
        assetHash[ symbol ] = ACT_HASH[ symbol ][ curDate ]

    MfdOptionsPrt.getProb = lambda self, x: x[ 'prob' ]
    
    with patch.object(MfdOptionsPrt, 'setMod', return_value=None):
        with patch.object(MfdOptionsPrt, 'setPrdDf', return_value=None):
            prtObj = MfdOptionsPrt(
                modFile      = None,
                assetHash    = assetHash,
                curDate      = curDate,
                minDate      = minDate,
                maxDate      = maxDate,
                minProb      = MIN_PROB,
                rfiDaily     = 0.0,
                tradeFee     = TRADE_FEE,
                nDayTimes    = 1140,
                logFileName  = None,                    
                verbose      = 0         )

    prtObj.logger = logger
    
    options = getOptions( curDate, optDf )
    
    selHash = prtObj.selOptions( 
        options,
        holdHash[ curDate ][ 'cash' ],
        MAX_PRICE_C,
        MAX_PRICE_A,
        NUM_DAILY_TRADES,
    )

    for item in selHash:

        for option in options:
            if option[ 'optionSymbol' ] == item:
                break

        cost = option[ 'contractCnt' ] * option[ 'unitPrice' ]
        
        logger.info(
            'Buying %d %s contract(s) at %0.2f!',
            selHash[ item ],
            item,
            cost,
        )
        
        holdHash[ curDate ][ 'cash' ] -= cost * selHash[ item ]
        holdHash[ curDate ][ 'options' ].append( option )

    return holdHash

def getOptionVal( option, curDate ):

    assetSymbol = option[ 'assetSymbol' ]
    oType       = option[ 'type' ]
    oCnt        = option[ 'contractCnt' ]
    strike      = float(option[ 'strike' ])
    uPrice      = option[ 'unitPrice' ]
    actPrice    = ACT_HASH[ assetSymbol ][ curDate ]

    cost = oCnt * uPrice
    
    if oType == 'call':
        actGain = oCnt * max(
            0.0,
            actPrice - strike
        )
    elif oType == 'put':
        actGain = oCnt * max(
            0.0,
            strike - actPrice
        )

    actRet = ( actGain - cost ) / cost
    
    return actGain, actRet

def settle( curDate, holdHash ):

    logger.info( 'Settling...' )
    
    for ind, option in enumerate(
            holdHash[ curDate ][ 'options' ]
    ):
        if option[ 'expiration' ] == curDate:
            gain, ret = getOptionVal( option, curDate )

            logger.info(
                'Exercising %s with a gain of %0.2f!',
                option[ 'optionSymbol' ],
                gain
            )
            
            holdHash[ curDate ][ 'options' ].pop( ind )
            holdHash[ curDate ][ 'cash' ] += gain 

    return holdHash

# ***********************************************************************
# Run the backtest
# ***********************************************************************

MfdOptionsPrt.getProb = lambda self, x: x[ 'prob' ]

retList = []
prevCash  = INIT_CASH
prevHolds = []

for curDate in dates:
    
    logger.info( 'Processing %s', str( curDate ) )

    holdHash[ curDate ][ 'cash' ] = prevCash
    holdHash[ curDate ][ 'options' ] = prevHolds
    
    logger.info(
        'Beginning the day with %0.2f cash and %d contracts!',
        prevCash,
        len( holdHash[ curDate ][ 'options' ] )
    )
    
    holdHash = selTradeOptions( curDate, holdHash, optDf )
    holdHash = settle( curDate, holdHash )

    balHash[ curDate ] = holdHash[ curDate ][ 'cash' ]
    
    for option in holdHash[ curDate ][ 'options' ]:
        actGain, actRet = getOptionVal( option, curDate )

        if option[ 'expiration' ] == curDate:
            retList.append( actRet )
    
        balHash[ curDate ] += actGain

    prevCash  = holdHash[ curDate ][ 'cash' ]
    prevHolds = holdHash[ curDate ][ 'options' ]
    
# ***********************************************************************
# Output
# ***********************************************************************

logger.info(
    'Mean / std of realized returns: %0.2f / %0.2f',
    np.mean( retList ),
    np.std( retList )
)

tmpList1 = []
tmpList2 = []
for item in balHash:
    tmpList1.append( item )
    tmpList2.append( balHash[ item ] )
    
outDf = pd.DataFrame(
    {
        'Date': tmpList1,
        'Balance': tmpList2,
    }
)

outDf[ 'Date' ] = outDf.Date.astype( 'datetime64[ns]' )

outDf[ 'Return' ] = outDf.Balance.pct_change()

logger.info(
    'Beginning / end balances: %0.2f / %0.2f',
    list( outDf.Balance )[0],
    list( outDf.Balance )[-1]
)

logger.info(
    'Mean / std / ratio of daily returns: %0.4f / %0.4f / %0.2f',
    outDf.Return.mean(),
    outDf.Return.std(),
    outDf.Return.mean() / outDf.Return.std()
)

outDf.to_csv( OUT_CSV_FILE, index = False )

plt.plot( outDf.Date, outDf.Balance )
plt.xlabel( 'Date' )
plt.ylabel( 'Portfolio Balance ($)' )
plt.show()

