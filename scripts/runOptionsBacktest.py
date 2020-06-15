# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import datetime
import pytz
import dill
import logging
import pickle
import numpy as np
import pandas as pd

sys.path.append( os.path.abspath( '../' ) )

import utl.utils as utl

from dat.assets import OPTION_ETFS as ETFS
from dat.assets import FUTURES, INDEXES
from mod.mfdMod import MfdMod
from prt.prt import MfdOptionsPrt

# ***********************************************************************
# Set some parameters 
# ***********************************************************************

optionDfFile = 'data/relevant_option_samples.pkl'
dfFile       = 'data/optionTestDfFile.pkl'
nTrnYears    = 5
nOosMinutes  = 5
maxHorizon   = 3 * 30
begDate      = pd.to_datetime( '2017-01-01' )
endDate      = pd.to_Datetime( '2018-12-31' )
numCores     = 2
INDEXES      = INDEXES + [ 'VIX' ]
ASSETS       = [ 'TLT', 'DIA', 'FAS', 'SMH' ]

# ***********************************************************************
# Get daily df to use for gettinf actual prices
# ***********************************************************************

df = pd.read_pickle( dfFile )

tmpFunc = lambda x : pd.to_datetime(x).stftime( '%Y-%m-%d' ) )

df[ 'Date0' ] = df.Date.apply( tmpFunc )

dayDf = df.groupby( 'Date', as_index = False ).mean()
                                                
# ***********************************************************************
# Utility functions
# ***********************************************************************

def getAssetHash( snapDate ):
    pass

def process( snapDate ):

    t0 = time.time()

    # Build model
    
    velNames = ETFS + INDEXES + FUTURES
    maxOosDt = snapDate
    maxTrnDt = maxOosDt - datetime.timedelta( minutes = nOosMinutes )
    minTrnDt = maxTrnDt - pd.DateOffset( years = nTrnYears )        
    modFile  = 'model_' + str( snapDate ) + '.dill'
    modFile  = os.path.join( 'models', modFile )
    
    mfdMod   = MfdMod( dfFile       = dfFile,
                       minTrnDate   = minTrnDt,
                       maxTrnDate   = maxTrnDt,
                       maxOosDate   = maxOosDt,
                       velNames     = velNames,
                       maxOptItrs   = 100,
                       optGTol      = 5.0e-2,
                       optFTol      = 5.0e-2,
                       regCoef      = 1.0e-3,
                       factor       = 1.0e-5,
                       logFileName  = None,
                       verbose      = 1      )

    sFlag = mfdMod.build()

    if not sFlag:
        logging.warning( 'Model did not converge!' )
        return False

    mfdMod.save( modFile )

    # Build portfolio object
    
    assetHash = getAssetHash( snapDate )
        
    prtObj = MfdOptionsPrt( modFile     = modFile,
                            assetHash   = assetHash,
                            curDate     = snapDate,
                            minDate     = minDate,
                            maxDate     = maxDate,
                            minProb     = 0.5,
                            rfiDaily    = 0.0,
                            tradeFee    = 0.75,
                            nDayTimes   = 1140,
                            logFileName = None,                    
                            verbose     = 1          )

    # Get historical options chains
    
    optDf = pd.read_pickle( optionDfFile )
    optDf = optDf[ optDf.UnderlyingSymbol.isin( ASSETS ) ]
    optDf = optDf[ optDf.Last > 0 ]
    optDf = optDf[ optDf.DataDate >= snapDate ]
        
    optDf[ 'horizon' ] = optDf[ 'Expiration' ] - optDf[ 'DataDate' ]
    optDf[ 'horizon' ] = optDf.horizon.apply( lambda x : x.days )

    optDf = optDf[ optDf.horizon <= maxHorizon ]

    # Get success probabilities of options
    
    probs = []

    optionSymbols = np.array( optDf.OptionSymbol )
    assetSymbols  = np.array( optDf.UnderlyingSymbol )
    strikes       = np.array( optDf.Strike )
    exprDates     = np.array( optDf.Expiration )
    unitPrices    = np.array( optDf.Last )
    optionTypes   = np.array( optDf.Type )
    
    for rowId in range( optDf.shape[0] ):

        option = { 'optionSymbol' : optionSymbols[rowId],
                   'assetSymbol'  : assetSymbols[rowId],
                   'strike'       : strikes[rowId],
                   'expiration'   : exprDates[rowId],
                   'contractCnt'  : 100,                     
                   'unitPrice'    : unitPrices[rowId],
                   'type'         : optionTypes[rowId],     }

        prob = prtObj.getProb( option )

        probs.append( prob )

    optDf[ 'Probability' ] = probs

    optFile  = 'options_' + str( snapDate ) + '.pkl'
    optFile  = os.path.join( 'models', optFile )
    
    optDf.to_pickle( optFile )
    
    return True

# ***********************************************************************
# Run the backtest
# ***********************************************************************

if __name__ ==  '__main__':
    
    snapDate = begDate
    pool     = Pool( numCores )

    while snapDate <= endDate:

        pool.apply_async( process, args = ( snapDate, ) )

        snapDate = snapDate + datetime.timedelta( days = maxHorizon )

    pool.close()
    pool.join()

    optFiles = os.listdir( 'models' )
    optDf = pd.DataFrame()
    
    for item in optFiles:

        if item.split( '_' )[0] != 'options':
            continue
    
        filePath = os.path.join( 'models', item )
        tmpDf = pd.read_pickle( filePath )
        optDf = pd.concat( [ optDf, tmpDf ] )

    optDf.to_pickle( 'all_options.pkl' )
