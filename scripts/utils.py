# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import datetime
import requests
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

# ***********************************************************************
# getPiDf(): Get a df from Pitrading data
# ***********************************************************************

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

# ***********************************************************************
# getDf(): Get a df from Qunadl and Pitrading
# ***********************************************************************

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

# ***********************************************************************
# getGeminiDf(): Get a df from Gemini csv files
# ***********************************************************************

def getGeminiDf( cryptoDir, velNames ):

    print( 'Getting intraday crypto currency data from gemni...' )

    t0    = time.time()
    df    = pd.DataFrame()

    for fileName in os.listdir( cryptoDir ):

        tmpList  = fileName.split( '.' )
        fileExt  = tmpList[-1]
        
        if fileExt != 'csv':
            continue

        tmpList  = fileName.split( '_' )

        if tmpList[0] != 'gemini':
            continue

        symbol   = tmpList[1][:3]

        if symbol not in velNames:
            continue

        filePath = os.path.join( cryptoDir, fileName )

        print( 'Reading', filePath )

        tmpDf = pd.read_csv( filePath, usecols = [ 'Date', 'Open' ] )

        tmpDf[ symbol ] = tmpDf.Open
        tmpDf = tmpDf[ [ 'Date', symbol ] ]

        if df.shape[0] == 0:
            df  = tmpDf
        else:
            if symbol in df.columns:
                df = pd.concat( [ df, tmpDf ] )
            else:
                df = df.merge( tmpDf, how = 'outer', on = [ 'Date' ] )

    if df.shape[0] > 0:
        df = df.sort_values( [ 'Date' ], ascending = [ True ] )
        df = df.reset_index( drop = True )
        df = df.interpolate( method = 'linear' )

    print( 'Done with getting intraday cryptocurrency data! ; Time =',
           round( time.time() - t0, 2 ) )

    return df

# ***********************************************************************
# getCryptoDf(): Get a df from Gemini (crypto) and Pitrading
# ***********************************************************************

def getCryptoDf( cryptoDir, piDir, velNames ):
    
    cDf = getGeminiDf( cryptoDir, velNames )
    pDf = getPiDf( piDir, velNames )

    if cDf.shape[0] > 0:
        cDf[ 'Date' ] = cDf.Date.apply( pd.to_datetime )

    if pDf.shape[0] > 0:
        dates = np.array( pDf[ 'Date' ] )
        times = np.array( pDf[ 'Time' ] )
        nRows = pDf.shape[0]

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

        pDf[ 'Date' ] = dates

    df = pd.DataFrame()

    if cDf.shape[0] > 0 and pDf.shape[0] > 0:
        df = cDf.merge( pDf, how = 'outer', on = [ 'Date' ] )
    elif cDf.shape[0] > 0:
        df = cDf
    elif pDf.shape[0] > 0:
        df = pDf

    if df.shape[0] > 0:
        df = df.sort_values( [ 'Date' ], ascending = [ True ] )
        df = df.reset_index( drop = True )
        df = df.interpolate( method = 'linear' )

    return df
    
# ***********************************************************************
# Some functions for getting cryto prices through cryptocompare.com
# ***********************************************************************

def minute_price_historical(symbol, comparison_symbol, limit, aggregate, exchange=''):
    url = 'https://min-api.cryptocompare.com/data/histominute?fsym={}&tsym={}&limit={}&aggregate={}'\
            .format(symbol.upper(), comparison_symbol.upper(), limit, aggregate)
    if exchange:
        url += '&e={}'.format(exchange)
    url += '&api_key={9f2079260740422a60206f9587882371b2eb2dd7f097969ab9e9d89ff8267173}'
    page = requests.get(url)
    data = page.json()['Data']
    df = pd.DataFrame(data)
    df['timestamp'] = [datetime.datetime.fromtimestamp(d) for d in df.time]
    return df

def hourly_price_historical(symbol, comparison_symbol, limit, aggregate, exchange=''):
    url = 'https://min-api.cryptocompare.com/data/histohour?fsym={}&tsym={}&limit={}&aggregate={}'\
            .format(symbol.upper(), comparison_symbol.upper(), limit, aggregate)
    if exchange:
        url += '&e={}'.format(exchange)
    page = requests.get(url)
    data = page.json()['Data']
    df = pd.DataFrame(data)
    df['timestamp'] = [datetime.datetime.fromtimestamp(d) for d in df.time]
    return df

def daily_price_historical(symbol, comparison_symbol, limit=1, aggregate=1, all_data=True, exchange=''):
    url = 'https://min-api.cryptocompare.com/data/histoday?fsym={}&tsym={}&limit={}&aggregate={}'\
            .format(symbol.upper(), comparison_symbol.upper(), limit, aggregate)
    if exchange:
        url += '&e={}'.format(exchange)
    if all_data:
        url += '&allData=true'
    page = requests.get(url)
    data = page.json()['Data']
    df = pd.DataFrame(data)
    df['timestamp'] = [datetime.datetime.fromtimestamp(d) for d in df.time]
    return df
