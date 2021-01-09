# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import datetime
import talib
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
from ode.odeGeo import OdeGeoConst
import ptc.ptc as ptc

# ***********************************************************************
# Main input params
# ***********************************************************************

BEG_DATE = pd.to_datetime( '2020-01-01 09:30:00' )
END_DATE = pd.to_datetime( '2021-01-06 15:30:00' )
MIN_TIME = '09:30:00'
MAX_TIME = '15:30:00'

NUM_TRN_DAYS = 360
NUM_OOS_DAYS = 3
NUM_PRD_MINS = 120

NUM_CORES = 2
DF_FILE   = 'data/dfFile_2020.pkl'
PTC_DIR   = 'pt_classifiers'

PTC_MIN_VIX = None
PTC_MAX_VIX = 60.0

VEL_NAMES = list( ETF_HASH.keys() ) + FUTURES
ASSETS    = list( SUB_ETF_HASH.keys() )

# ***********************************************************************
# Some utils
# ***********************************************************************
        
def buildMod( snapDate ):

    maxOosDt    = snapDate
    maxTrnDt    = maxOosDt - datetime.timedelta( days = NUM_OOS_DAYS )
    minTrnDt    = maxTrnDt - datetime.timedelta( days = NUM_TRN_DAYS )

    mfdMod = MfdMod( dfFile       = DF_FILE,
                     minTrnDate   = minTrnDt,
                     maxTrnDate   = maxTrnDt,
                     maxOosDate   = maxOosDt,
                     velNames     = VEL_NAMES,
                     maxOptItrs   = 500,
                     optGTol      = 1.0e-3,
                     optFTol      = 1.0e-3,
                     regCoef      = 5.0e-3,
                     factor       = 4.0e-05,
                     logFileName  = None,
                     verbose      = 1          )
        
    sFlag = mfdMod.build()

    assert sFlag, 'Model build was unsuccessful!'

    return mfdMod.ecoMfd

def getGeoSlopes( ecoMfd ):

    nDims     = ecoMfd.nDims
    actOosSol = ecoMfd.actOosSol
    Gamma     = ecoMfd.getGammaArray( ecoMfd.GammaVec )
    bcVec     = np.zeros( shape = ( nDims ), dtype = 'd' )

    for m in range( ecoMfd.nDims ):
        bcVec[m] = actOosSol[m][-1] 

    odeObj = OdeGeoConst( Gamma    = Gamma,
                          bcVec    = bcVec,
                          bcTime   = 0.0,
                          timeInc  = 1.0,
                          nSteps   = NUM_PRD_MINS - 1,
                          intgType = 'LSODA',
                          tol      = 1.0e-2,
                          verbose  = True       )

    sFlag    = odeObj.solve()

    assert sFlag, 'Geodesic equation did not converge!'
    
    prdSol = odeObj.getSol()
    
    slopeHash = {}
    
    for m in range( ecoMfd.nDims ):
        asset     = ecoMfd.velNames[m]
        tmp       = ecoMfd.deNormHash[ asset ]
        slope     = tmp[0]
        intercept = tmp[1]
            
        for i in range( NUM_PRD_MINS ):
            prdSol[m][i] = slope * prdSol[m][i] + intercept

        if prdSol.shape[0] != ecoMfd.nDims:
            assert False, 'Inconsistent prdSol size!'

        if prdSol.shape[1] <= 0:
            assert False, 'Number of minutes should be positive!'
        
        slopeHash[ asset ] = prdSol[m][1] - prdSol[m][0]

    return slopeHash

def getPerfs( ecoMfd ):
    
    perfs = ecoMfd.getOosTrendPerfs()
    
    perfHash = {}
    
    for m in range( ecoMfd.nDims ):
        
        asset = ecoMfd.velNames[m]
        
        if perfs[m]:
            perfHash[ asset ] = 1
        else:
            perfHash[ asset ] = 0

    return perfHash
    
def getMacdTrends( ecoMfd ):

    nDims     = ecoMfd.nDims
    nTimes    = ecoMfd.nTimes
    nOosTimes = ecoMfd.nOosTimes
    actSol    = ecoMfd.actSol
    actOosSol = ecoMfd.actOosSol
    nTmp      = nTimes + nOosTimes - 1
    tmpVec    = np.empty( shape = ( nTmp ), dtype = 'd' )
    
    macdHash  = {}

    for m in range( nDims ):
        asset     = ecoMfd.velNames[m]
        tmp       = ecoMfd.deNormHash[ asset ]
        slope     = tmp[0]
        intercept = tmp[1]

        for i in range( nTimes ):
            tmpVec[i] = slope * actSol[m][i] + intercept

        for i in range( 1, nOosTimes ):
            tmpVec[i + nTimes - 1] = slope * actOosSol[m][i] + intercept
    
        macd, signal, hist = talib.MACD(
            tmpVec, 
            fastperiod   = 12, 
            slowperiod   = 26,
            signalperiod = 9
        ) 

        macdHash[ asset ] = np.sign( macd[-1] - signal[-1] )

    return macdHash

def getMsdRatios( ecoMfd ):

    nDims     = ecoMfd.nDims
    nTimes    = ecoMfd.nTimes
    nOosTimes = ecoMfd.nOosTimes
    actSol    = ecoMfd.actSol
    actOosSol = ecoMfd.actOosSol
    nTmp      = nTimes + nOosTimes - 1
    tmpVec    = np.empty( shape = ( nTmp ), dtype = 'd' )
    msdHash   = {}

    for m in range( nDims ):
        asset     = ecoMfd.velNames[m]
        tmp       = ecoMfd.deNormHash[ asset ]
        slope     = tmp[0]
        intercept = tmp[1]

        for i in range( nTimes ):
            tmpVec[i] = slope * actSol[m][i] + intercept

        for i in range( 1, nOosTimes ):
            tmpVec[i + nTimes - 1] = slope * actOosSol[m][i] + intercept

        meanVal = tmpVec.mean()
        stdVal  = tmpVec.std()
        currVal = tmpVec[-1]
        
        msdHash[ asset ] = ( currVal - meanVal ) / stdVal

    return msdHash

def getAcls( snapDate ):

    dayDf = pd.read_pickle( DF_FILE )        

    dayDf[ 'Date' ] = dayDf.Date.astype( 'datetime64[ns]' )
    
    minDate = snapDate - \
        pd.DateOffset( days = 7 )
    
    dayDf = dayDf[ ( dayDf.Date >= minDate ) &
                   ( dayDf.Date <= snapDate ) ]
    
    dayDf[ 'Date' ] = dayDf.Date.\
        apply( lambda x : x.strftime( '%Y-%m-%d' ) )
    
    dayDf = dayDf.groupby( 'Date', as_index = False ).mean()

    dayDf[ 'Date' ] = dayDf.Date.astype( 'datetime64[ns]' )
        
    dayDf = dayDf.sort_values( [ 'Date' ], ascending = True )

    aclHash = {}
    
    for symbol in ASSETS:

        dayDf[ 'vel' ] = np.gradient( dayDf[ symbol ], 2 )
        dayDf[ 'acl' ] = np.gradient( dayDf[ 'vel' ], 2 )

        symVal = list( dayDf.acl )[-1] 

        aclHash[ symbol ] = symVal

    return aclHash

def buildPtc( snapDate ):

    invAssets = [ SUB_ETF_HASH[ item ] for item in ASSETS ]

    allAssets = ASSETS + invAssets
    
    for symbol in allAssets:

        ptcObj = ptc.PTClassifier( symbol      = symbol,
                                   symFile     = 'data/%s.pkl' % symbol,
                                   vixFile     = 'data/VIX.pkl',
                                   ptThreshold = 1.0e-2,
                                   nPTAvgDays  = None,
                                   testRatio   = 0,
                                   method      = 'bayes',
                                   minVix      = None,
                                   maxVix      = 60.0,
                                   minTrnDate  = None,
                                   maxTrnDate  = snapDate,
                                   logFileName = None,                    
                                   verbose     = 1          )

        ptcObj.classify()
        
        ptcFile = os.path.join( PTC_DIR,
                                'ptc_' + symbol + '.pkl' )

        ptcObj.save( ptcFile )

def getPtcTags( snapDate ):
    
    buildPtc( snapDate )
    
    dayDf = pd.read_pickle( DF_FILE )        

    dayDf[ 'Date' ] = dayDf.Date.astype( 'datetime64[ns]' )
    
    minDate = snapDate - \
        pd.DateOffset( days = 7 )
    
    dayDf = dayDf[ ( dayDf.Date >= minDate ) &
                   ( dayDf.Date <= snapDate ) ]
    
    dayDf[ 'Date' ] = dayDf.Date.\
        apply( lambda x : x.strftime( '%Y-%m-%d' ) )
    
    dayDf = dayDf.groupby( 'Date', as_index = False ).mean()
    
    dayDf[ 'Date' ] = dayDf.Date.astype( 'datetime64[ns]' )
        
    dayDf = dayDf.sort_values( [ 'Date' ], ascending = True )

    vixVal = list( dayDf.VIX )[-1]

    vixFlag = False

    if PTC_MIN_VIX is not None and vixVal < PTC_MIN_VIX:
        vixFlag = True
    if PTC_MAX_VIX is not None and vixVal > PTC_MAX_VIX:
        vixFlag = True
        
    ptcHash = {}
    
    for symbol in ASSETS:

        invSymbol = ETF_HASH[ symbol ]
        
        dayDf[ 'vel' ] = np.gradient( dayDf[ symbol ], 2 )
        dayDf[ 'acl' ] = np.gradient( dayDf[ 'vel' ], 2 )
        fea1 = list( dayDf.acl )[-1]
        ptcFile = os.path.join( PTC_DIR,
                                'ptc_' + symbol + '.pkl' )
        obj = pickle.load( open( ptcFile, 'rb' ) )
        X = np.array( [ [ fea1 ] ] )
        ptTag = obj.predict( X )[0]

        dayDf[ 'invVel' ] = np.gradient( dayDf[ invSymbol ], 2 )
        dayDf[ 'invAcl' ] = np.gradient( dayDf[ 'invVel' ], 2 )
        invFea1 = list( dayDf.invAcl )[-1]
        invPtcFile = os.path.join( PTC_DIR,
                                'ptc_' + invSymbol + '.pkl' )
        invObj = pickle.load( open( invPtcFile, 'rb' ) )        
        X = np.array( [ [ invFea1 ] ] )
        invPtTag = invObj.predict( X )[0]

        if vixFlag:
            ptcHash[ symbol ] = 0
        elif ptTag == ptc.PEAK:
            ptcHash[ symbol ] = 1
        elif invPtTag == ptc.PEAK:
            ptcHash[ symbol ] = -1
        else:
            ptcHash[ symbol ] = 0
    
    return ptcHash    

def run( snapDate ):

    ecoMfd    = buildMod( snapDate )
    slopeHash = getGeoSlopes( ecoMfd )
    perfHash  = getPerfs( ecoMfd )
    macdHash  = getMacdTrends( ecoMfd )
    msdHash   = getMsdRatios( ecoMfd )
    aclHash   = getAcls( snapDate )
    ptcHash   = getPtcTags( snapDate )

    geoTrHash = {}
    for symbol in slopeHash:
        geoTrHash[ symbol ] = np.sign( slopeHash[ symbol ] )

    outHash = {
        'symbol': [],
        'geo_slope': [],
        'geo_trend': [],
        'geo_perf': [],
        'macd_trend': [],
        'msd_ratio': [],
        'acl': [],
        'ptc_tag': [],
    }

    for symbol in ASSETS:
        outHash[ 'symbol' ].append( symbol )
        outHash[ 'geo_slope' ].append( slopeHash[ symbol ] )
        outHash[ 'geo_trend' ].append( geoTrHash[ symbol ] )
        outHash[ 'geo_perf' ].append( perfHash[ symbol ] )
        outHash[ 'macd_trend' ].append( macdHash[ symbol ] )
        outHash[ 'msd_ratio' ].append( msdHash[ symbol ] )
        outHash[ 'acl' ].append( aclHash[ symbol ] )
        outHash[ 'ptc_tag' ].append( ptcHash[ symbol ] )

    outDf = pd.DataFrame( outHash )

    outDf.to_csv(
        'models/NNfeature_%s.csv' % str( snapDate ),
        index = False
    )

# ***********************************************************************
# Run 
# ***********************************************************************

if __name__ ==  '__main__':
    
    snapDate = BEG_DATE
    pool     = Pool( NUM_CORES )

    while snapDate <= END_DATE:

        while True:
            if snapDate.isoweekday() not in [ 6, 7 ] and \
               snapDate.strftime( '%H:%M:%S' ) >= MIN_TIME and \
               snapDate.strftime( '%H:%M:%S' ) <= MAX_TIME:
                break
            else:
                snapDate += datetime.timedelta( minutes = NUM_PRD_MINS )

        pool.apply_async( run, args = ( snapDate, ) )

        snapDate = snapDate + datetime.timedelta( minutes = NUM_PRD_MINS )

    pool.close()
    pool.join()
    
    modFiles = os.listdir( 'models' )

    outDf = pd.DataFrame()
    
    for item in modFiles:

        if item.split( '_' )[0] != 'NNfeature':
            continue
    
        filePath = os.path.join( 'models', item )
        tmpDf    = pd.read_csv( filePath )
        outDf    = np.concat( [ outDf, tmpDf ] )

    outDf.to_csv( 'models/NNfeatures_all.csv', index = False )
