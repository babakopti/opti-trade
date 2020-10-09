# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import datetime
import dill
import pickle
import logging
import json
import numpy as np
import pandas as pd

from multiprocessing import Process, Pool

sys.path.append( os.path.abspath( '../' ) )

import utl.utils as utl

from dat.assets import SUB_ETF_HASH, ETF_HASH
from dat.assets import FUTURES
from mod.mfdMod import MfdMod
from prt.prt import MfdPrt 

# ***********************************************************************
# Main input params
# ***********************************************************************

prtFile     = 'portfolios/test.json'
bkBegDate   = pd.to_datetime( '2020-08-01 09:30:00' )
bkEndDate   = pd.to_datetime( '2020-09-09 15:30:00' )
nTrnDays    = 360
nOosDays    = 3
nPrdMinutes = 3 * 60
minModTime  = '09:30:00'
maxModTime  = '15:30:00'

# ***********************************************************************
# Set some parameters and read data
# ***********************************************************************

modFlag  = False
dataFlag = False
numCores = 3

baseDir  = '/var/data'
dfFile   = 'data/dfFile_2020.pkl'

# symbols  = list( ETF_HASH.keys() ) + \
#            list( ETF_HASH.values() ) + \
#            FUTURES + \
#            [ 'VIX' ]
velNames  = list( ETF_HASH.keys() ) + FUTURES
assetPool = list( SUB_ETF_HASH.keys() )

factor = 4.0e-05
vType  = 'vel'

# ***********************************************************************
# build data file if needed
# ***********************************************************************

if dataFlag:
    nDays   = nTrnDays + nOosDays + 7
    minDate = bkBegDate - datetime.timedelta( days = nDays )
    
    dataDf = utl.mergeSymbols( symbols = symbols,
                               datDir  = baseDir,
                               fileExt = 'pkl',
                               minDate = minDate,
                               logger  = None     )
    
    dataDf.to_pickle( dfFile )
    
# ***********************************************************************
# Utility functions
# ***********************************************************************

def buildModPrt( snapDate ):

    maxOosDt    = snapDate
    maxTrnDt    = maxOosDt - datetime.timedelta( days = nOosDays )
    minTrnDt    = maxTrnDt - datetime.timedelta( days = nTrnDays )
    modFilePath = 'models/model_' + str( snapDate ) + '.dill'
    wtFilePath  = 'models/weights_' + str( snapDate ) + '.pkl'

    if modFlag:

        t0     = time.time()

        mfdMod = MfdMod( dfFile       = dfFile,
                         minTrnDate   = minTrnDt,
                         maxTrnDate   = maxTrnDt,
                         maxOosDate   = maxOosDt,
                         velNames     = velNames,
                         maxOptItrs   = 500,
                         optGTol      = 1.0e-3,
                         optFTol      = 1.0e-3,
                         regCoef      = 5.0e-3,
                         factor       = factor,
                         logFileName  = None,
                         verbose      = 1          )
        
        sFlag = mfdMod.build()

        if not sFlag:
            logging.warning( 'Warning: Model build was unsuccessful!' )
            logging.warning( 'Warning: Not building a portfolio based on this model!!' )
            return False

        mfdMod.save( modFilePath )
    else:
        try:
            mfdMod = dill.load( open( modFilePath, 'rb' ) )
        except Exception as exc:
            logging.critical(exc)
            sys.exit()

    print( 'Building portfolio for snapdate', snapDate )

    t0     = time.time()
    wtHash = {}
#    curDt  = snapDate
#    endDt  = snapDate + datetime.timedelta( days = nPrdDays )
#    nDays  = ( endDt - curDt ).days

    nPrdTimes = nPrdMinutes #int( nDays * 19 * 60 )
    nRetTimes = int( 30 * 19 * 60 )  

    assetList = assetPool
    
    # assetList = utl.filtDeviation( assetPool,
    #                               dfFile,
    #                               snapDate,
    #                               nAvgDays  = 7,
    #                               nCurDays  = 1,
    #                               threshold = 2.0, 
    #                               logger    = None    )

    eDf = utl.sortAssets( symbols   = assetList,
                          dfFile    = dfFile,
                          begDate   = snapDate - datetime.timedelta( days = 60 ),
                          endDate   = snapDate,
                          criterion = 'abs_sharpe',                          
                          mode      = 'daily'     )
    assets = list( eDf.asset )[:5]

    ecoMfd = mfdMod.ecoMfd
    quoteHash = {}
    for m in range( ecoMfd.nDims ):

        asset = ecoMfd.velNames[m]

        if asset not in assets:
            continue
        
        tmp       = ecoMfd.deNormHash[ asset ]
        slope     = tmp[0]
        intercept = tmp[1]
        
        quoteHash[ asset ] = slope * ecoMfd.actOosSol[m][-1] + intercept
        
    mfdPrt = MfdPrt( modFile      = modFilePath,
                     quoteHash    = quoteHash,
                     nRetTimes    = nRetTimes,
                     nPrdTimes    = nPrdTimes,
                     strategy     = 'equal',
                     minProbLong  = 0.5,
                     minProbShort = 0.5,
                     vType        = vType,
                     fallBack     = 'macd',
                     verbose      = 1          )

    dateKey = snapDate.strftime( '%Y-%m-%d %H:%M:00' )

    tmpHash = mfdPrt.getPortfolio()

    # for symbol in tmpHash:

    #     df = pd.read_pickle( dfFile )

    #     df = df[ [ 'Date', symbol ] ]

    #     df[ 'Date' ] = df.Date.apply( pd.to_datetime )
        
    #     tmpDate = snapDate - pd.DateOffset( days = 60 )
        
    #     df = df[ (df.Date >= tmpDate) & (df.Date <= snapDate) ]

    #     df[ 'Date0' ] = df.Date.apply( lambda x : x.strftime( '%Y-%m-%d' ) )

    #     dayDf = df.groupby( 'Date0', as_index = False )[ symbol ].mean()
    #     dayDf = dayDf.rename( columns = { 'Date0' : 'Date' } )

    #     dayDf[ 'vel' ] = np.gradient( dayDf[ symbol ], 2 )
    #     dayDf[ 'acl' ] = np.gradient( dayDf[ 'vel' ], 2 )

    #     dayDf[ 'avgAcl' ] = dayDf.acl.rolling( min_periods = 1,
    #                                            window = 7 ).mean()
    #     dayDf[ 'feature' ] = dayDf.acl - dayDf.avgAcl        
                 
    #     obj = pickle.load( open( 'pt_classifiers/ptc_%s.pkl' % symbol, 'rb' ) )

    #     dayDf = dayDf.sort_values( [ 'Date' ], ascending = True )
        
    #     val = list( dayDf.feature )[-1]

    #     X = np.array( [ val ] ).reshape( ( 1, 1 ) )
        
    #     ptTag = obj.predict( X )[0]

    #     # if ptTag == 1:
    #     #     tmpHash[ symbol ] = -abs( tmpHash[ symbol ] )
    #     if ptTag == 2:
    #         tmpHash[ symbol ] = abs( tmpHash[ symbol ] )

    wtHash[ dateKey ] = tmpHash
    
    pickle.dump( wtHash, open( wtFilePath, 'wb' ) )    

    print( 'Building portfolio took %d seconds!' % ( time.time() - t0 ) )

    return True

# ***********************************************************************
# Run the backtest
# ***********************************************************************

if __name__ ==  '__main__':
    
    snapDate = bkBegDate
    pool     = Pool( numCores )

    while snapDate <= bkEndDate:

        while True:
            if snapDate.isoweekday() not in [ 6, 7 ] and \
               snapDate.strftime( '%H:%M:%S' ) >= minModTime and \
               snapDate.strftime( '%H:%M:%S' ) <= maxModTime:
                break
            else:
                snapDate += datetime.timedelta( minutes = nPrdMinutes )

        pool.apply_async( buildModPrt, args = ( snapDate, ) )

        snapDate = snapDate + datetime.timedelta( minutes = nPrdMinutes )

    pool.close()
    pool.join()
    
    modFiles = os.listdir( 'models' )

    prtWtsHash = {}

    for item in modFiles:

        if item.split( '_' )[0] != 'weights':
            continue
    
        filePath = os.path.join( 'models', item )
        tmpHash = pickle.load( open( filePath, 'rb' ) )
        dateStr = list( tmpHash.keys() )[0]
        prtWtsHash[ dateStr ] = tmpHash[ dateStr ]

    with open( prtFile, 'w' ) as fp:
            json.dump( prtWtsHash, fp )        
