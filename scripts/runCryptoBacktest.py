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

from dat.assets import INDEXES
from mod.mfdMod import MfdMod
from prt.prt import MfdPrt 

# ***********************************************************************
# Main input params
# ***********************************************************************

prtFile     = 'portfolios/crypto_24_hours_no_short_9PM.json'
bkBegDate   = pd.to_datetime( '2020-01-01 21:00:00' )
bkEndDate   = pd.to_datetime( '2020-10-27 21:00:00' )
nTrnDays    = 360
nOosDays    = 3
nPrdMinutes = 24 * 60
minModTime  = '00:00:00'
maxModTime  = '23:59:00'

# ***********************************************************************
# Set some parameters and read data
# ***********************************************************************

modFlag  = True
numCores = 4

dfFile   = 'data/dfFile_crypto.pkl'

velNames  = [ 'BTC', 'ETH', 'LTC', 'ZEC' ] + INDEXES + [ 'VIX' ]
assetPool = [ 'BTC', 'ETH', 'LTC', 'ZEC' ]

factor = 4.0e-05
vType  = 'vel'

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
                         optGTol      = 1.0e-5,
                         optFTol      = 1.0e-5,
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

    nPrdTimes = nPrdMinutes 
    nRetTimes = int( 30 * 19 * 60 )  

    assets = assetPool

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

    # No short sells
    for symbol in tmpHash:
        tmpHash[ symbol ] = max( 0.0, tmpHash[ symbol ] )
        
    sumAbs = sum( [abs(x) for x in tmpHash.values()] )
    sumAbsInv = 1.0
    if sumAbs > 0:
        sumAbsInv = 1.0 / sumAbs

    for symbol in tmpHash:
        tmpHash[ symbol ] = sumAbsInv * tmpHash[ symbol ]
    
    wtHash[ dateKey ] = tmpHash
    
    pickle.dump( wtHash, open( wtFilePath, 'wb' ) )    

    print( 'Building portfolio took %d seconds!' % ( time.time() - t0 ) )

    os.remove( modFilePath )
    
    return True

# ***********************************************************************
# Run the backtest
# ***********************************************************************

if __name__ ==  '__main__':
    
    snapDate = bkBegDate
    pool     = Pool( numCores )

    while snapDate <= bkEndDate:

        while True:
            if snapDate.strftime( '%H:%M:%S' ) >= minModTime and \
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
