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
import gc
import numpy as np
import pandas as pd

from multiprocessing import Pool

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
snapDates    = [ pd.to_datetime( '2016-05-02 10:00:00' ),
                 pd.to_datetime( '2016-05-16 10:00:00' ),                 
                 pd.to_datetime( '2016-05-27 10:00:00' ),
                 pd.to_datetime( '2016-09-01 10:00:00' ),
                 pd.to_datetime( '2016-09-15 10:00:00' ),
                 pd.to_datetime( '2016-09-30 10:00:00' ),
                 pd.to_datetime( '2017-07-03 10:00:00' ),
                 pd.to_datetime( '2017-07-14 10:00:00' ),                 
                 pd.to_datetime( '2017-07-31 10:00:00' )  ]

optDf = pd.read_pickle( optionDfFile )

snapDates = list( set( optDf.DataDate ) - set( snapDates ) )
                 
INDEXES      = INDEXES 
ASSETS       = [ 'TLT', 'DIA', 'FAS', 'SMH' ]
velNames     = ETFS + INDEXES + FUTURES
velNames     = list( set( velNames ) - { 'TRAN', 'TYX', 'OIH' } )

# ***********************************************************************
# Do some preprocessing
# ***********************************************************************

actDf = pd.read_pickle( dfFile )[ [ 'Date' ] + velNames ]

actDf[ 'Date' ] = actDf.Date.apply( lambda x : \
                                    pd.to_datetime(x).\
                                    strftime( '%Y-%m-%d' ) )

actDf = actDf.groupby( 'Date', as_index = False ).mean()

actHash = {}
for date in actDf.Date:
    actHash[ date ] = {}
    tmpDf = actDf[ actDf.Date == date ]
    for symbol in velNames:
        actHash[ date ][ symbol ] = list( tmpDf[ symbol ] )[0]

if False:        
    actDf = actDf.melt( id_vars    = [ 'Date' ],
                        value_vars = list( set( actDf.columns ) - \
                                           { 'Date' } ) )

    actDf[ 'Date' ] = actDf.Date.apply( lambda x : pd.to_datetime(x) )

    actDf = actDf.rename( columns = { 'Date'     : 'Expiration',
                                      'variable' : 'UnderlyingSymbol',
                                      'value'    : 'actExprPrice' }   )

    optDf = pd.read_pickle( optionDfFile )

    optDf[ 'Expiration' ] = optDf.Expiration.apply( lambda x : pd.to_datetime(x) )
    optDf[ 'DataDate' ]   = optDf.DataDate.apply( lambda x : pd.to_datetime(x) )
    
    optDf = optDf.merge( actDf,
                         how = 'left',
                         on  = [ 'UnderlyingSymbol', 'Expiration' ] )

    optDf.to_pickle( optionDfFile )

# ***********************************************************************
# Utility functions
# ***********************************************************************

def getAssetHash( snapDate ):
    
    return actHash[ snapDate.strftime( '%Y-%m-%d' ) ]
    
def process( snapDate ):

    t0 = time.time()

    # Build model

    print( 'Building model for %s' % str( snapDate ) )
    
    maxOosDt = snapDate
    maxTrnDt = maxOosDt - datetime.timedelta( minutes = nOosMinutes )
    minTrnDt = maxTrnDt - pd.DateOffset( years = nTrnYears )        
    modFile  = 'model_' + str( snapDate ) + '.dill'
    modFile  = os.path.join( 'models', modFile )

    print( maxOosDt, maxTrnDt, minTrnDt )

    mfdMod   = MfdMod( dfFile       = dfFile,
                       minTrnDate   = minTrnDt,
                       maxTrnDate   = maxTrnDt,
                       maxOosDate   = maxOosDt,
                       velNames     = velNames,
                       maxOptItrs   = 100,
                       optGTol      = 1.0e-3,
                       optFTol      = 1.0e-3,
                       regCoef      = 1.0e-3,
                       factor       = 1.0e-5,
                       logFileName  = None,
                       verbose      = 1      )

    sFlag = mfdMod.build()

    if not sFlag:
        print( 'Model did not converge!' )

    mfdMod.save( modFile )

    mfdMod = None
    gc.collect()
    
    # Build portfolio object

    print( 'Getting assetHash for %s' % str( snapDate ) )
    
    assetHash = getAssetHash( snapDate )

    print( 'Building prt obj for %s' % str( snapDate ) )

    minDate = snapDate
    maxDate = snapDate + datetime.timedelta( days = maxHorizon )
    prtObj  = MfdOptionsPrt( modFile     = modFile,
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

    print( 'Adding success probabilities option df for %s' % str( snapDate ) )
    
    optDf = pd.read_pickle( optionDfFile )
    optDf = optDf[ optDf.UnderlyingSymbol.isin( ASSETS ) ]
    optDf = optDf[ optDf.Last > 0 ]
    optDf = optDf[ optDf.DataDate >= snapDate ]
    optDf = optDf[ ( optDf.Expiration >= minDate ) &
                   ( optDf.Expiration <= maxDate ) ]
        
    optDf[ 'horizon' ] = optDf[ 'Expiration' ] - optDf[ 'DataDate' ]

    optDf = optDf[ optDf.horizon <= datetime.timedelta( days = maxHorizon ) ]

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

    print( 'Saving opt df for %s' % str( snapDate ) )
    
    optDf.to_pickle( optFile )
    
    return True

# ***********************************************************************
# Run the backtest
# ***********************************************************************

if __name__ ==  '__main__':

    for snapDate in snapDates:

        process( snapDate )

    optFiles = os.listdir( 'models' )
    optDf = pd.DataFrame()
    
    for item in optFiles:

        if item.split( '_' )[0] != 'options':
            continue
    
        filePath = os.path.join( 'models', item )
        tmpDf = pd.read_pickle( filePath )
        optDf = pd.concat( [ optDf, tmpDf ] )

    optDf.to_pickle( 'all_options.pkl' )
