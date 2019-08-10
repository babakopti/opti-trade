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
# getDf(): Get a df from data
# ***********************************************************************

def getDf( quandlDir, piDir, velNames ):

    t0 = time.time()

    print( 'Getting macros...' )

    dataPaths = []

    tmpFlag = True
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

        if tmpFlag:
            qDf = tmpDf
            tmpFlag = False
        else:
            qDf = qDf.merge( tmpDf, how = 'outer', on = [ 'Date' ] )

    qDf = qDf.interpolate( method = 'linear' )
    qDf = qDf.dropna()
    qDf = qDf.reset_index()

    qDf[ 'Date' ] = qDf.Date.apply( lambda x:datetime.datetime.strptime( x, '%Y-%m-%d' ) )

    print( 'Done with getting macros!; Time =', time.time() - t0 )

    t0 = time.time()

    print( 'Getting intraday data...' )

    tmpFlag = True

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

        tmpDf[ fileBase ] = tmpDf.Open
        tmpDf = tmpDf[ [ 'Date', 'Time', fileBase ] ]

        if tmpFlag:
            pDf = tmpDf
            tmpFlag = False
        else:
            pDf = pDf.merge( tmpDf, how = 'outer', on = [ 'Date', 'Time' ] )

    pDf = pDf.interpolate( method = 'linear' )
    pDf = pDf.dropna()
    pDf = pDf.reset_index()

    pDf[ 'Date' ] = pDf.Date.apply( lambda x:datetime.datetime.strptime( x, '%m/%d/%Y' ) )

    print( 'Done with getting intraday data! ; Time =',
           time.time() - t0 )

    t0 = time.time()

    print( 'Merging all data...' )

    df = pDf.merge( qDf, how = 'left', on = [ 'Date' ] )

    df = df.interpolate( method = 'linear' )
    
    df = df.dropna()
    df = df.reset_index()
    df = df[ [ 'Date', 'Time' ] + velNames ]

    print( df.head() )

    print( 'Done with merging all data! ; Time =',
           time.time() - t0 )

    return df
