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
import scipy
from scipy.optimize import line_search
import matplotlib.pyplot as plt

sys.path.append( os.path.abspath( '../' ) )

import utl.utils as utl

from dat.assets import ETF_HASH

# ***********************************************************************
# Set some parameters
# ***********************************************************************

dataFlag = False
prtFlag  = False

baseDir  = '/var/data'
prtDir   = '/var/prt_weights'
dfFile   = 'exposure_opt_dfFile.pkl'
prtFile  = 'exposure_opt_prtWtsHash.json'
symbols  = list( ETF_HASH.keys() ) + \
           list( ETF_HASH.values() ) + \
           [ 'VIX' ]
minDate  = pd.to_datetime( '2020-02-15' )
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

def getPrtWtsHash():
    
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
        
        wtHash = json.load( open( os.path.join( prtDir, fileName ), 'r' ) )
            
        prtWtsHash[ snapDate ] = wtHash

    return prtWtsHash

# ***********************************************************************
# get prt wts hash
# ***********************************************************************

if prtFlag:
    prtWtsHash = getPrtWtsHash()
    with open( prtFile, 'w' ) as fp:
            json.dump( prtWtsHash, fp )
else:
    prtWtsHash = json.load( open( prtFile, 'r' ) )
    
# ***********************************************************************
# getVix()
# ***********************************************************************

def getVixHash():

    wtDf = pd.DataFrame( { 'Date': list( prtWtsHash.keys() ) } )
    df = pd.read_pickle( dfFile )
    df = df[ [ 'Date', 'VIX' ] ]
    
    df[ 'Date' ] = df.Date.apply( lambda x: x.strftime( '%Y-%m-%d %H:%M:%S' ) )
    
    df = df.merge( wtDf, how = 'outer', on = 'Date' )
    df = df.interpolate( method = 'linear' )
    
    vixHash = {}
    dates   = list( df.Date )
    vixes   = list( df.VIX )

    for i in range( len( dates ) ):
        vixHash[ dates[i] ] = vixes[i] 

    return vixHash

# ***********************************************************************
# Storing vix and prt wts hash 
# ***********************************************************************

vixHash = getVixHash()

# ***********************************************************************
# getExprReturn
# ***********************************************************************

def getExprHash( coefs ):

    assert len( coefs ) == 4, 'Incorrect input size!'

    longExprCoefs  = coefs[:2]
    shortExprCoefs = coefs[2:]

    longHash  = {}
    shortHash = {}

    for snapDate in prtWtsHash:

        wtHash = prtWtsHash[ snapDate ]
        vix    = vixHash[ snapDate ]

        longExpr = longExprCoefs[0] - longExprCoefs[1] * np.tanh( vix )

        shortExpr = shortExprCoefs[0] - shortExprCoefs[1] * np.tanh( vix )        

        longHash[ snapDate ]  = longExpr
        shortHash[ snapDate ] = shortExpr

    return longHash, shortHash

# ***********************************************************************
# getObjFunc
# ***********************************************************************

def getObjFunc( coefs ):

    longHash, shortHash = getExprHash( coefs )

    for snapDate in prtWtsHash:

        wtHash = prtWtsHash[ snapDate ]

        longExpr = longHash[ snapDate ]
        shortExpr = shortHash[ snapDate ]
            
        for item in wtHash:
            if wtHash[ item ] > 0:
                wtHash[ item ] = longExpr * wtHash[ item ]
            elif wtHash[ item ] < 0:
                wtHash[ item ] = shortExpr * wtHash[ item ]                
    
    retDf = utl.calcBacktestReturns( prtWtsHash = prtWtsHash,
                                     dfFile     = dfFile,
                                     initTotVal = initTotVal,
                                     shortFlag  = False,
                                     invHash    = ETF_HASH,
                                     hourOset   = 6   )
    plt.plot(retDf.Date, retDf.EndVal)
    
    return 1.0 - float( retDf.Return.mean() )
    
# # ***********************************************************************
# # Optimize
# # ***********************************************************************

if False:
    options  = { 'ftol'       : 0.001,
                 'maxiter'    : 100,
                 'disp'       : True  }

    optObj = scipy.optimize.minimize( fun         = getObjFunc, 
                                      x0          = [ 1.0, 0., 1.0, 1.0 ], 
                                      method      = 'SLSQP',
                                      bounds      = [ (0.0, None),
                                                      (0.0, None),
                                                      (0.0, None),
                                                      (0.0, None) ],
                                      options     = options    )

    print( 'Success:', optObj.success )
    
    print( optObj.x )


if False:
    coefs = np.linspace(0,0.01,10)
    y = []
    for coef in coefs:
        y.append(1.0 - getObjFunc( [ 1.0, 0.0, 1.0, coef ] ))

    plt.plot( coefs, y, '-o' )
    plt.show()

print( 'Full exposure average daily return:', 1.0 - getObjFunc( [ 1.0, 0.0, 1.0, 0.0 ] ) )
plt.show()


