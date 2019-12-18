# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import datetime
import requests
import logging
import numpy as np
import pandas as pd

from logging.handlers import SMTPHandler
from io import StringIO

# ***********************************************************************
# getLogger(): Get a logger object
# ***********************************************************************

def getLogger( logFileName, verbose, pkgName = None ):
    
    verboseHash = { 0 : logging.NOTSET,
                    1 : logging.INFO,
                    2 : logging.DEBUG }
        
    logger      = logging.getLogger( pkgName )
        
    logger.setLevel( verboseHash[ verbose ] )
        
    if logFileName is None:
        fHd = logging.StreamHandler() 
    else:
        fHd = logging.FileHandler( logFileName )

    logFmt = logging.Formatter( '%(asctime)s - %(name)s %(levelname)-s - %(message)s' )
        
    fHd.setFormatter( logFmt )
        
    logger.addHandler( fHd )

    return logger

# ***********************************************************************
# getLogger(): Get a logger object
# ***********************************************************************

def getAlertHandler( alertLevel, subject = None, mailList = [] ):
    
    mHd = SMTPHandler( mailhost    = ( 'smtp.gmail.com', 587 ),
                       fromaddr    = 'optilive.noreply@gmail.com',
                       toaddrs     = mailList,
                       subject     = subject,
                       credentials = ('optilive.noreply@gmail.com',
                                      'optilivenoreply'),
                       secure      = () )
    
    mHd.setLevel( alertLevel )
        
    return mHd

# ***********************************************************************
# getKibotAuth(): Athenticate Kibot to get data
# ***********************************************************************

def getKibotAuth( timeout = 60, maxTries = 10 ):

    username = 'babak.emami@gmail.com'
    password = 'a1aba1aba'

    url  = 'http://api.kibot.com?action=login&user=%s&password=%s' \
        % ( username, password )
        
    for itr in range( maxTries ):
        try:
            resp = requests.get( url, timeout = timeout )
            break
        except:
            time.sleep( 5 )
            continue

    return resp.ok

# ***********************************************************************
# getKibotData( ): Read data from kibot
# ***********************************************************************

def getKibotData( etfs     = [],
                  futures  = [],
                  stocks   = [],
                  indexes  = [],
                  nDays    = 365,
                  interval = 1,
                  maxTries = 10,
                  timeout  = 60,
                  logger   = None   ):

    t0       = time.time()
    initFlag = True

    # Set the logger object is None
    
    if logger is None:
        logger = getLogger( None, 1 )

    # Set a hash table of asset types
    
    typeHash = {}

    for symbol in etfs:
        typeHash[ symbol ] = 'ETFs'

    for symbol in futures:
        typeHash[ symbol ] = 'futures'

    for symbol in stocks:
        typeHash[ symbol ] = 'stocks'

    for symbol in indexes:
        typeHash[ symbol ] = 'indexes'        

    # Get a data frae of intraday ETFs, stocks, and futures
    
    symbols = etfs + futures + stocks
    
    for symbol in symbols:

        logger.info( 'Reading symbol %s...' % symbol )

        authFlag = getKibotAuth()
            
        if not authFlag:
            logger.error( 'Authentication failed!' )
            assert False, 'Authentication failed!'
    
        url  = 'http://api.kibot.com/?action=history'
        url  = url + '&symbol=%s&interval=%d&period=%d&type=%s&regularsession=1' \
            % ( symbol, interval, nDays, typeHash[ symbol ] )

        for itr in range( maxTries ):
            try:
                resp = requests.get( url, timeout = timeout )
                break
            except:
                time.sleep( 5 )
                continue

        if not resp.ok:
            logger.error( 'Query for %s failed!', symbol )
            assert False, 'Query for %s failed!' % symbol
    
        cols = [ 'Date',
                 'Time',
                 'Open',
                 'High',
                 'Low',
                 'Close',
                 'Volume' ]
    
        tmpDf = pd.read_csv( StringIO( resp.text ), names = cols )
        tmpDf = tmpDf.rename( columns = { 'Open' : symbol } )
        tmpDf = tmpDf[ [ 'Date', 'Time', symbol ] ]

        if initFlag:
            df       = tmpDf
            initFlag = False
        else:
            df = df.merge( tmpDf,
                           how = 'outer',
                           on  = [ 'Date', 'Time' ] )

    df = df[ [ 'Date', 'Time' ] + symbols ]
    df = df.interpolate( method = 'linear' )
    df = df.dropna()
    df = df.reset_index( drop = True )

    # Get a data frame of daily indexes
    
    initFlag = True
    
    for symbol in indexes:

        logger.info( 'Reading symbol %s...' % symbol )

        authFlag = getKibotAuth()
            
        if not authFlag:
            logger.error( 'Authentication failed!' )
            assert False, 'Authentication failed!'

        url  = 'http://api.kibot.com/?action=history&symbol=$%s&interval=daily&period=%d' \
            % ( symbol, nDays )

        for itr in range( maxTries ):
            try:
                resp = requests.get( url, timeout = timeout )
                break
            except:
                time.sleep( 5 )
                continue

        if not resp.ok:
            logger.error( 'Query for %s failed!', symbol )
            assert False, 'Query for %s failed!' % symbol
    
        cols = [ 'Date',
                 'Open',
                 'High',
                 'Low',
                 'Close',
                 'Volume' ]
    
        tmpDf = pd.read_csv( StringIO( resp.text ), names = cols )
        tmpDf = tmpDf.rename( columns = { 'Open' : symbol } )
        tmpDf = tmpDf[ [ 'Date', symbol ] ]

        if initFlag:
            indDf    = tmpDf
            initFlag = False
        else:
            indDf = indDf.merge( tmpDf,
                                 how = 'outer',
                                 on  = [ 'Date' ] )
            
    indDf = indDf[ [ 'Date' ] + indexes ]
    
    # Merge intraday and daily data frames
    
    tmpList = np.unique( list( df[ 'Date' ] ) )
    tmpHash = {}
    
    for date in tmpList:
        tmpDf = df[ df.Date == date ]
        tmpDf = tmpDf.sort_values( [ 'Time' ],
                                   ascending = [ True ] )
        tmpHash[ date ] = np.min( tmpDf.Time )

    tmpFunc = lambda x : tmpHash[ x ]

    indDf[ 'Time' ] = indDf.Date.apply( tmpFunc )

    df = df.merge( indDf,
                   how = 'left',
                   on  = [ 'Date', 'Time' ] )
    
    df = df[ [ 'Date', 'Time' ] + symbols + indexes ]
    df = df.interpolate( method = 'linear' )
    df = df.dropna()
    df = df.reset_index( drop = True )

    # Combine and Date and Time columns
    
    dates = np.array( df[ 'Date' ] )
    times = np.array( df[ 'Time' ] )
    nRows = df.shape[0]

    assert len( dates ) == nRows, 'Inconsistent size of dates!'
    assert len( times ) == nRows, 'Inconsistent size of times!'

    logger.info( 'Formatting Date column...' )
    
    for i in range( nRows ):

        tmpStr  = str( dates[i] )
        tmpList = tmpStr.split( '/' )

        assert len( tmpList ) == 3, 'Wrong date format!'
        
        year    = int( tmpList[2] )
        month   = int( tmpList[0] )
        day     = int( tmpList[1] )        
        
        tmpStr  = str( times[i] )
        tmpList = tmpStr.split( ':' )

        assert len( tmpList ) == 2, 'Wrong time format!'
        
        hour    = int( tmpList[0] )
        minute  = int( tmpList[1] )
        second  = 0

        date = datetime.datetime( year,
                                  month,
                                  day,
                                  hour,
                                  minute,
                                  second ).strftime( '%Y-%m-%d %H:%M:%S' )
        df.iloc[ i,0 ] = date
        
    df = df[ [ 'Date' ] + symbols + indexes ]
    df = df.sort_values( [ 'Date' ], ascending = [ True ] )
    df = df.reset_index( drop = True )

    logger.info( 'Getting %d symbols took %0.2f seconds!',
                 len( symbols + indexes ), 
                 time.time() - t0 )

    return df
    
