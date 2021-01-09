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

VEL_NAMES = list( ETF_HASH.keys() ) + FUTURES
ASSETS    = list( SUB_ETF_HASH.keys() )

ALL_DF = pd.read_pickle( DF_FILE )

# ***********************************************************************
# Some utils
# ***********************************************************************
        
def getActTrend( snapDate, nHours ):

    minDate = snapDate
    maxDate = snapDate + datetime.timedelta( hours = nHours )

    df = ALL_DF[ ( ALL_DF.Date >= minDate ) & ( ALL_DF.Date <= maxDate ) ]

    trendHash = {}
    
    for symbol in ASSETS:
        vec = list( df[ symbol ] )
        trendHash[ symbol ] = np.sign( vec[-1] - vec[0] )

    return trendHash

def run( snapDate ):

    trendHash1  = getActTrend( snapDate, 1 )
    trendHash2  = getActTrend( snapDate, 2 )
    trendHash3  = getActTrend( snapDate, 3 )
    trendHash6  = getActTrend( snapDate, 6 )
    trendHash12 = getActTrend( snapDate, 12 )
    trendHash24 = getActTrend( snapDate, 24 )

    outHash = {
        'symbol': [],
        'act_trend_1': [],
        'act_trend_2': [],
        'act_trend_3': [],
        'act_trend_6': [],
        'act_trend_12': [],
        'act_trend_24': [],
    }

    for symbol in ASSETS:
        outHash[ 'symbol' ].append( symbol )
        outHash[ 'act_trend_1' ].append( trendHash1[ symbol ] )
        outHash[ 'act_trend_2' ].append( trendHash2[ symbol ] )
        outHash[ 'act_trend_3' ].append( trendHash3[ symbol ] )
        outHash[ 'act_trend_6' ].append( trendHash6[ symbol ] )
        outHash[ 'act_trend_12' ].append( trendHash12[ symbol ] )
        outHash[ 'act_trend_24' ].append( trendHash24[ symbol ] )        

    outDf = pd.DataFrame( outHash )

    outDf.to_csv(
        'models/NNresps_%s.csv' % str( snapDate ),
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

        if item.split( '_' )[0] != 'NNresps':
            continue
    
        filePath = os.path.join( 'models', item )
        tmpDf    = pd.read_csv( filePath )
        outDf    = np.concat( [ outDf, tmpDf ] )

    outDf.to_csv( 'models/NNresps_all.csv', index = False )
