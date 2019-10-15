# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import datetime
import numpy as np
import pandas as pd

# ***********************************************************************
# Some definitions
# ***********************************************************************

MIN_PI_ROWS = 1000000

# ***********************************************************************
# getQuandlDf(): Get a df from quandl data
# ***********************************************************************

def getQuandlDf( quandlDir, velNames ):

    print( 'Getting macros...' )

    t0    = time.time()
    qDf   = pd.DataFrame()
    qFlag = False

    for fileName in os.listdir( quandlDir ):

        tmpList  = fileName.split( '.' )
        fileExt  = tmpList[-1]

        if fileExt not in [ 'csv', 'pkl' ]:
            continue

        filePath = os.path.join( quandlDir, fileName )
        fileBase = '.'.join( tmpList[:-1] )

        if fileBase not in velNames:
            continue

        if fileExt == 'csv':
            tmpDf = pd.read_csv( filePath, usecols = [ 'Date', fileBase ] )
        elif fileExt == 'pkl':
            tmpDf = pd.read_pickle( filePath )
        else:
            assert False, 'Unknown file %s' % filePath

        if not qFlag:
            qDf   = tmpDf
            qFlag = True
        else:
            qDf = qDf.merge( tmpDf, how = 'outer', on = [ 'Date' ] )

    if qFlag:
        qDf = qDf.interpolate( method = 'linear' )
        qDf = qDf.dropna()
        qDf = qDf.reset_index( drop = True )

        qDf[ 'Date' ] = qDf.Date.apply( lambda x:datetime.datetime.strptime( x, '%Y-%m-%d' ) )

    print( 'Done with getting macros!; Time =', round( time.time() - t0, 2 ) )

    return qDf

def getPiDf( piDir, velNames ):

    print( 'Getting intraday data...' )

    t0    = time.time()
    pDf   = pd.DataFrame()
    pFlag = False

    for fileName in os.listdir( piDir ):

        tmpList  = fileName.split( '.' )
        fileExt  = tmpList[-1]
        
        if fileExt not in [ 'csv', 'pkl', 'zip' ]:
            continue

        filePath = os.path.join( piDir, fileName )
        fileBase = '.'.join( tmpList[:-1] )

        if fileBase not in velNames:
            continue

        print( 'Reading', filePath )

        if fileExt in [ 'csv', 'zip' ]:
            tmpDf = pd.read_csv( filePath, usecols = [ 'Date', 'Time', 'Open' ] )
        elif fileExt == 'pkl':
            tmpDf = pd.read_pickle( filePath )
        else:
            assert False, 'Unknown file %s' % filePath

        if tmpDf.shape[0] < MIN_PI_ROWS:
            print( 'Dropping', fileBase, 'nRows =', tmpDf.shape[0] ) 
            continue

        tmpDf[ fileBase ] = tmpDf.Open
        tmpDf = tmpDf[ [ 'Date', 'Time', fileBase ] ]

        if not pFlag:
            pDf   = tmpDf
            pFlag = True
        else:
            pDf = pDf.merge( tmpDf, how = 'outer', on = [ 'Date', 'Time' ] )
            pDf = pDf.sort_values( [ 'Date', 'Time' ], ascending = [ True, True ] )
            pDf = pDf.reset_index( drop = True )

    if pFlag:
        pDf[ 'Date' ] = pDf.Date.apply( lambda x:datetime.datetime.strptime( x, '%m/%d/%Y' ) )
        pDf = pDf.sort_values( [ 'Date' ], ascending = [ True ] )
        pDf = pDf.reset_index( drop = True )
        pDf = pDf.interpolate( method = 'linear' )

    print( 'Done with getting intraday data! ; Time =',
           round( time.time() - t0, 2 ) )

    return pDf

def getDf( quandlDir, piDir, velNames ):
    
    qDf = getQuandlDf( quandlDir, velNames )
    df  = getPiDf( piDir, velNames )
    if qDf.shape[0] > 0 and df.shape[0] > 0:
        df = df.merge( qDf, how = 'left', on = [ 'Date' ] )
    elif qDf.shape[0] > 0:
        df = qDf

    df.reset_index( drop = True )

    dates          = np.array( df[ 'Date' ] )
    times          = np.array( df[ 'Time' ] )
    nRows          = df.shape[0]
        
    for i in range( nRows ):

        tmpStr = str( times[i] )
        nTmp   = len( tmpStr )

        if  nTmp < 4:
            nTmp1  = 4 - nTmp
            tmpStr = ''.join( ['0'] * nTmp1 ) + tmpStr

        hour     = int( tmpStr[:2] )
        minute   = int( tmpStr[2:] )
        dates[i] = dates[i] +\
            np.timedelta64( hour,   'h' ) +\
            np.timedelta64( minute, 'm' )

    df[ 'Date' ] = dates

    return df
    
