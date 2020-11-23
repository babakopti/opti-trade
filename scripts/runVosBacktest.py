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

OUT_CSV_FILE = 'portfolios/vos_max_cost_50_max_hz_10.csv'
OPTION_CHAIN_FILE = 'options_chain_data/option_chain_Aug_Nov_2020.pkl'
ACT_FILE = 'data/dfFile_2020-11-10 15:00:06.pkl'

INIT_CASH = 300.0

MIN_PROB  = 0.49
MAX_PAIR_COST  = 50.0
MAX_UNIQUE_PAIR_COUNT = 1
TRADE_FEE = 2 * 0.65
MIN_HORIZON = 1
MAX_HORIZON = 10
MAX_TRIES = 1000
MAX_DAILY_CASH = 50.0

BID_FACTOR = 0.85

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
ASSET_HASH = {}
for symbol in set( optDf.assetSymbol ):
    ACT_HASH[ symbol ] = dict( zip( actDf[ 'Date' ], actDf[ symbol ] ) )
    ASSET_HASH[ symbol ] = None
    
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
        columns = {
            'Prob' : 'prob',
            'unitPrice': 'unitPriceAsk',
        }
    )
    tmpDf[ 'unitPriceBid' ] = tmpDf.unitPriceAsk.apply(
        lambda x: BID_FACTOR * x
    )
    tmpDf = tmpDf[ [ 'optionSymbol',
                     'assetSymbol',
                     'type',
                     'contractCnt',
                     'strike',
                     'expiration',
                     'unitPriceAsk',
                     'unitPriceBid',                     
                     'prob'     ] ]
    
    options = [
        dict( item ) for ind, item in tmpDf.iterrows()
    ]

    return options

def selTradeOptions( curDate, holdHash, optDf ):
    
    minDate = curDate + datetime.timedelta( days = MIN_HORIZON )
    maxDate = curDate + datetime.timedelta( days = MAX_HORIZON )

    with patch.object(MfdOptionsPrt, 'setMod', return_value=None):
        with patch.object(MfdOptionsPrt, 'setPrdDf', return_value=None):
            prtObj = MfdOptionsPrt(
                modFile      = None,
                assetHash    = ASSET_HASH,
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

    cash = min( MAX_DAILY_CASH, holdHash[ curDate ][ 'cash' ] )
    
    selList = prtObj.getVosPortfolio(
        options,
        cash,
        MAX_PAIR_COST,
        MAX_UNIQUE_PAIR_COUNT,
        MAX_TRIES,
    )
    
    if selList is None:
        logger.info( 'Ran out of money! Stopping trading!' )
        return holdHash

    for item in selList:

        pairHash = item[0]
        
        for itr in range( item[1] ):
            holdHash[ curDate ][ 'cash' ] -= pairHash[ 'cost' ]
            holdHash[ curDate ][ 'options' ].append( pairHash )

    return holdHash

def getOptionsPairVal( pairHash, curDate ):

    assetSymbol = pairHash[ 'assetSymbol' ]
    oType       = pairHash[ 'type' ]
    oCnt        = pairHash[ 'contractCnt' ]
    strikeBuy   = pairHash[ 'strikeBuy' ]
    strikeSell  = pairHash[ 'strikeSell' ]
    cost        = pairHash[ 'cost' ]
    actPrice    = ACT_HASH[ assetSymbol ][ curDate ]

    if oType == 'call':
        actGain = oCnt * max(
            0.0,
            min( actPrice, strikeSell ) - strikeBuy
        )
    elif oType == 'put':
        actGain = oCnt * max(
            0.0,
            strikeBuy - max( actPrice, strikeSell )
        )

    actRet = ( actGain - cost ) / cost
    
    return actGain, actRet

def settle( curDate, holdHash ):

    logger.info( 'Settling...' )
    
    for ind, pairHash in enumerate(
            holdHash[ curDate ][ 'options' ]
    ):
        exprDate = pd.to_datetime( pairHash[ 'expiration' ] )
        
        if exprDate == curDate:
            gain, ret = getOptionsPairVal( pairHash, curDate )

            logger.info(
                'Exercising %s/%s contracts with a gain of %0.2f!',
                pairHash[ 'optionSymbolBuy' ],
                pairHash[ 'optionSymbolSell' ],
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
    
    for pairHash in holdHash[ curDate ][ 'options' ]:
        actGain, actRet = getOptionsPairVal( pairHash, curDate )

        exprDate = pd.to_datetime( pairHash[ 'expiration' ] )
        if exprDate == curDate:
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

