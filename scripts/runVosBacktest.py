# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import datetime
import logging
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from unittest.mock import patch

sys.path.append( os.path.abspath( '..' ) )

import utl.utils as utl
import ptc.ptc as ptc

from prt.prt import MfdOptionsPrt 

# ***********************************************************************
# Main input params
# ***********************************************************************

OUT_BAL_FILE = 'portfolios/vos_max_cost_25_max_hz_10.csv'
OUT_HIST_FILE = 'data/vos_max_cost_25_max_hz_10_hist.csv'
OPTION_CHAIN_FILE = 'options_chain_data/option_chain_Aug_Nov_2020.pkl'
ACT_FILE = 'data/dfFile_2020-11-10 15:00:06.pkl'

INIT_CASH = 300.0

MIN_PROB  = 0.48
MAX_PAIR_COST  = 25.0
MAX_UNIQUE_PAIR_COUNT = 2
TRADE_FEE = 0.65
MIN_HORIZON = 1
MAX_HORIZON = 10
MAX_TRIES = 1000
MAX_DAILY_CASH = 25.0

BID_FACTOR = 0.8

PTC_FLAG      = False
PTC_MIN_VIX   = None
PTC_MAX_VIX   = 60.0
PTC_HEAD      = 'ptc_'
PTC_DIR       = 'pt_classifiers'
BASE_DAT_DIR  = 'data'

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
# Initialize the holding hash and history hash
# ***********************************************************************

holdHash = {}
balHash = {}
HIST_HASH = {}

for date in dates:
    
    holdHash[ date ] = {
        'cash': None,
        'options': [],
    }

# ***********************************************************************
# Some utility functions
# ***********************************************************************

def buildPTC( symbols ):

    if not PTC_FLAG:
        return
        
    for symbol in symbols:

        symFile = os.path.join( BASE_DAT_DIR,
                                '%s.pkl' % symbol )
        vixFile = os.path.join( BASE_DAT_DIR,
                                'VIX.pkl' )            

        ptcObj  = ptc.PTClassifier( symbol      = symbol,
                                    symFile     = symFile,
                                    vixFile     = vixFile,
                                    ptThreshold = 1.0e-2,
                                    nPTAvgDays  = None,
                                    testRatio   = 0,
                                    method      = 'bayes',
                                    minVix      = PTC_MIN_VIX,
                                    maxVix      = PTC_MAX_VIX,
                                    logFileName = None,                    
                                    verbose     = 1          )

        ptcObj.classify()

        ptcFile = os.path.join( PTC_DIR,
                                PTC_HEAD + symbol + '.pkl' )
            
        logger.info( 'Saving the classifier to %s', ptcFile )
            
        ptcObj.save( ptcFile )

def getPTCBlacklists( curDate, assetSymbols ):

    if not PTC_FLAG:
        return None

    logger.info( 'Applying peak classifiers to portfolio!' )
        
    dayDf = pd.read_pickle( ACT_FILE )        

    dayDf[ 'Date' ] = dayDf.Date.astype( 'datetime64[ns]' )
    
    minDate = curDate - \
        pd.DateOffset( days = 7 )
    
    dayDf = dayDf[ ( dayDf.Date >= minDate ) &
                   ( dayDf.Date <= curDate ) ]
    
    dayDf[ 'Date' ] = dayDf.Date.\
            apply( lambda x : x.strftime( '%Y-%m-%d' ) )
    
    dayDf = dayDf.groupby( 'Date', as_index = False ).mean()

    dayDf[ 'Date' ] = dayDf.Date.astype( 'datetime64[ns]' )
        
    dayDf = dayDf.sort_values( [ 'Date' ], ascending = True )

    vixVal = list( dayDf.VIX )[-1]

    if PTC_MIN_VIX is not None and vixVal < PTC_MIN_VIX:
        logger.critical( 'Did not use PTC as current VIX of '
                         '%0.2f is not in range!',
                         vixVal )
        return None

    if PTC_MAX_VIX is not None and vixVal > PTC_MAX_VIX:
        logger.critical( 'Did not use PTC as current VIX of '
                         '%0.2f is not in range!',
                         vixVal )
        return None

    callBlackList = []
    putBlackList  = []
    
    for symbol in assetSymbols:
        
        dayDf[ 'vel' ] = np.gradient( dayDf[ symbol ], 2 )
        dayDf[ 'acl' ] = np.gradient( dayDf[ 'vel' ], 2 )

        symVal = list( dayDf.acl )[-1] 
            
        ptcFile = os.path.join( PTC_DIR,
                                PTC_HEAD + symbol + '.pkl' )
            
        obj = pickle.load( open( ptcFile, 'rb' ) )

        X = np.array( [ [ symVal ] ] )
        
        ptTag = obj.predict( X )[0]

        if ptTag == ptc.PEAK:
            
            logger.critical( 'A peak is detected for %s!', symbol )

            callBlackList.append( symbol )
            
        elif ptTag == ptc.TROUGH:
            
            logger.critical( 'A trough is detected for %s!',
                             symbol )
            
            putBlackList.append( symbol )

    return {
        'callBlackList': callBlackList,
        'putBlackList': putBlackList
    }

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

    callBlackList = None
    putBlackList  = None
    
    if PTC_FLAG:
        blackListHash = getPTCBlacklists(
            curDate,
            set( optDf.assetSymbol ),
        )

        callBlackList = blackListHash[ 'callBlackList' ]
        putBlackList  = blackListHash[ 'putBlackList' ]
        
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
        callBlackList,
        putBlackList,
    )
    
    if selList is None:
        logger.info( 'Ran out of money! Stopping trading!' )
        return holdHash

    for item in selList:

        pairHash = item[0]

        pairSymbol = '/'.join( [
            pairHash[ 'optionSymbolBuy' ],
            pairHash[ 'optionSymbolSell' ],
        ] )
        
        for itr in range( item[1] ):
            holdHash[ curDate ][ 'cash' ] -= pairHash[ 'cost' ]
            holdHash[ curDate ][ 'options' ].append( pairHash )
            
        HIST_HASH[ pairSymbol ] = \
            {
                'buyDate': curDate,
                'cost': pairHash[ 'cost' ],
                'count': item[1],
            }

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

    remainList = []
    
    for pairHash in holdHash[ curDate ][ 'options' ]:

        exprDate = pd.to_datetime( pairHash[ 'expiration' ] )
        
        if exprDate == curDate:
            
            gain, ret = getOptionsPairVal( pairHash, curDate )

            pairSymbol = '/'.join( [
                pairHash[ 'optionSymbolBuy' ],
                pairHash[ 'optionSymbolSell' ],
            ] )
            
            logger.info(
                'Exercising %s pair with a gain of %0.2f!',
                pairSymbol,
                gain
            )
            
            holdHash[ curDate ][ 'cash' ] += gain

            HIST_HASH[ pairSymbol ][ 'exercise' ] = curDate
            HIST_HASH[ pairSymbol ][ 'gain' ] = gain            

        else:
            remainList.append( pairHash )
            
        holdHash[ curDate ][ 'options' ] = remainList
        
    return holdHash

# ***********************************************************************
# Build PTC models
# ***********************************************************************

if PTC_FLAG:
    buildPTC( set( optDf.assetSymbol ) )

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
# Put together and output history
# ***********************************************************************

tmpList = []
for pairSymbol in HIST_HASH:
    tmpList.append( {
        **{ 'pairSymbol': pairSymbol },
        **HIST_HASH[ pairSymbol ],
    } )

histDf = pd.DataFrame( tmpList )

histDf.to_csv( OUT_HIST_FILE, index = False )

# ***********************************************************************
# Output and plot balance
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

outDf.to_csv( OUT_BAL_FILE, index = False )

plt.plot( outDf.Date, outDf.Balance )
plt.xlabel( 'Date' )
plt.ylabel( 'Portfolio Balance ($)' )
plt.show()

