# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import json
import re
import datetime
import pytz
import numpy as np
import pandas as pd

from google.cloud import storage

sys.path.append( os.path.abspath( '../' ) )

import utl.utils as utl

from dat.assets import SUB_ETF_HASH

# ***********************************************************************
# Import libraries
# ***********************************************************************

dataFlag = True
baseDir  = '/var/data'
prtDir   = '/var/prt_weights'
dfFile   = 'exposure_opt_dfFile.pkl' 
symbols  = list( SUB_ETF_HASH.keys() ) + list( SUB_ETF_HASH.values() )
minDate  = pd.to_datetime( '2020-01-22' )
maxDate  = pd.to_datetime( '2020-05-08' )

initTotVal  = 20000

# ***********************************************************************
# Import libraries
# ***********************************************************************

if dataFlag:
    dataDf = utl.mergeSymbols( symbols = symbols,
                               datDir  = baseDir,
                               fileExt = 'pkl',
                               minDate = minDate,
                               logger  = None     )
    
    dataDf = dataDf[ dataDf.Date <= maxDate ]

    dataDf.to_pickle( dfFile )

# ***********************************************************************
# getPrtWtsHash()
# ***********************************************************************

def getPrtWtsHash( longExprCoefs, shortExprCoefs ):

    df = pd.read_pickle( dfFile )
    df = df[ [ 'Date', 'VIX' ] ]
    
    vixHash = {}
    dates   = list( df.Date )
    vixes   = list( df.VIX )
    
    for i in range( len( dates ) ):
        vixHash[ dates[i] ] = vixes[i]
    
    prtWtsHash = {}

    pattern = 'prt_weights_\d+-\d+-\d+_\d+:\d+:\d+.json'
    
    for fileName in os.listdir( prtDir ):

        if not re.search( pattern, fileName ):
            continue

        dateList = re.findall( '\d+', fileName )
        year     = int( dateList[0] )
        month    = int( dateList[1] )
        day      = int( dateList[2] )
        hour     = int( dateList[3] )
        minute   = int( dateList[4] )
        second   = 0
        snapDate = datetime.datetime( year, month,  day,
                                      hour, minute, second )

        if snapDate < minDate:
            continue

        if snapDate > maxDate:
            continue

        snapDate = snapDate.strftime( '%Y-%m-%d %H:%M:%S' )
        
        wtHash = json.load( os.path.join( prtDir, fileName ) )

        vix = vixHash[ snapDate ]

        longCoef = 0.0
        for k in range( len( longExprCoefs ) ):
            longCoef += longExprCoefs[k] * vix**k
            
        shortCoef = 0.0
        for k in range( len( shortExprCoefs ) ):
            shortCoef += shortExprCoefs[k] * vix**k
            
        for item in wtHash:
            if wtHash[ item ] > 0:
                wtHash[ item ] = longCoef * wtHash[ item ]
            elif wtHash[ item ] < 0:
                wtHash[ item ] = shortCoef * wtHash[ item ]                
            
        wtHash = defaultdict( float, wtHash )
        
        prtWtsHash[ snapDate ] = wtHash

    return prtWtsHash

# ***********************************************************************
# getReturns
# ***********************************************************************

def getReturns( longExprCoefs = [ 1.0 ], shortExprCoefs = [ 1.0 ] ):

    prtWtsHash = getPrtWtsHash( longExprCoefs, shortExprCoefs )
    
    retDf = utl.calcBacktestReturns( prtWtsHash = prtWtsHash,
                                     dfFile     = dfFile,
                                     initTotVal = initTotVal,
                                     shortFlag  = False,
                                     invHash    = SUB_ETF_HASH   )

    return float( retDf.Return.mean() )
    
    
