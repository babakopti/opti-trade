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
from collections import defaultdict

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

    USER_NAME = 'apikey'
    API_KEY = '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'
    
    mHd = SMTPHandler( mailhost    = ( 'smtp.sendgrid.net', 587 ),
                       fromaddr    = 'optilive.noreply@gmail.com',
                       toaddrs     = mailList,
                       subject     = subject,
                       credentials = ( USER_NAME, API_KEY ),
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
# combineDateTime( ): Combine Date and Time columns
# ***********************************************************************

def combineDateTime( df ):
    
    assert 'Date' in df.columns, 'Date column not found!'
    assert 'Time' in df.columns, 'Time column not found!'    

    df[ 'Date' ] = ( df[ 'Date' ] + \
                     ' ' + \
                     df[ 'Time' ] ).apply( pd.to_datetime )
    
    cols = set( df.columns ) - set( [ 'Date', 'Time' ] )
    cols = [ 'Date' ] + list( cols )
    df   = df[ cols ]
    df   = df.sort_values( [ 'Date' ], ascending = [ True ] )
    
    return df

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
                  minRows  = 100,
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
    df      = pd.DataFrame()

    if len( symbols + indexes ) == 0:
        logger.warning( 'No symbol is given!' )
        return None
    
    for symbol in symbols:

        logger.info( 'Reading symbol %s...' % symbol )

        authFlag = getKibotAuth()
            
        if not authFlag:
            logger.error( 'Authentication failed!' )
            assert False, 'Authentication failed!'
    
        url  = 'http://api.kibot.com/?action=history'
        url  = url + '&symbol=%s&interval=%d&period=%d&type=%s&regularsession=0' \
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
        nRows = tmpDf.shape[0]
        
        logger.info( 'Got %d rows for %s!', nRows, symbol )

        if nRows < minRows:
            logger.warning( resp.text )
            logger.warning( 'Skipping %s as it has only %d rows!',
                            symbols,
                            nRows  )
            continue
            
        if initFlag:
            df       = tmpDf
            initFlag = False
        else:
            df = df.merge( tmpDf,
                           how = 'outer',
                           on  = [ 'Date', 'Time' ] )
            
    if df.shape[0] > 0:
        df = df[ [ 'Date', 'Time' ] + symbols ]
        df = df.reset_index( drop = True )

    # Get a data frame of daily indexes
    
    initFlag = True
    indDf    = pd.DataFrame()
    
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
        tmpDf = tmpDf.rename( columns = { 'Close' : symbol } )
        tmpDf = tmpDf[ [ 'Date', symbol ] ]
        nRows = tmpDf.shape[0]
        
        logger.info( 'Got %d rows for %s!', nRows, symbol )

        if initFlag:
            indDf    = tmpDf
            initFlag = False
        else:
            indDf = indDf.merge( tmpDf,
                                 how = 'outer',
                                 on  = [ 'Date' ] )
    if len( indexes ) > 0:
        indDf = indDf[ [ 'Date' ] + indexes ]

        if len( symbols ) == 0:
            df = indDf
    
    # Merge intraday and daily data frames

    if len( symbols ) > 0 and len( indexes ) > 0:
        tmpList = set( list( df[ 'Date' ] ) )
        tmpHash = defaultdict( lambda: '23:59' )
    
        for date in tmpList:
            tmpDf = df[ df.Date == date ]
            tmpDf = tmpDf.sort_values( [ 'Time' ],
                                       ascending = [ True ] )
            tmpHash[ date ] = np.max( tmpDf.Time )

        tmpFunc = lambda x : tmpHash[ x ]

        indDf[ 'Time' ] = indDf.Date.apply( tmpFunc )

        df = df.merge( indDf,
                       how = 'left',
                       on  = [ 'Date', 'Time' ] )
    
        df = df[ [ 'Date', 'Time' ] + symbols + indexes ]

    if len( symbols ) > 0:
        df = combineDateTime( df )
    elif len( indexes ) > 0:
        df[ 'Date' ] = df[ 'Date' ].apply( pd.to_datetime )
        
    df = df.sort_values( [ 'Date' ], ascending = [ True ] )    
    df = df.interpolate( method = 'linear' )
    df = df.dropna()
    df = df.reset_index( drop = True )
    
    logger.info( 'Getting %d symbols took %0.2f seconds!',
                 len( symbols + indexes ), 
                 time.time() - t0 )
    
    return df

# ***********************************************************************
# getDailyKibotData( ): Read daily data from kibot
# ***********************************************************************

def getDailyKibotData( etfs     = [],
                       futures  = [],
                       stocks   = [],
                       indexes  = [],
                       nDays    = 365,
                       maxTries = 10,
                       timeout  = 60,
                       minRows  = 3,
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
    
    symbols = etfs + futures + stocks + indexes
    df      = pd.DataFrame()

    if len( symbols ) == 0:
        logger.warning( 'No symbol is given!' )
        return None
    
    for symbol in symbols:

        logger.info( 'Reading symbol %s...' % symbol )

        authFlag = getKibotAuth()
            
        if not authFlag:
            logger.error( 'Authentication failed!' )
            assert False, 'Authentication failed!'
    
        url  = 'http://api.kibot.com/?action=history'
        url  = url + '&symbol=%s&interval=daily&period=%d&type=%s&regularsession=0' \
            % ( symbol, nDays, typeHash[ symbol ] )

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
        nRows = tmpDf.shape[0]
        
        logger.info( 'Got %d rows for %s!', nRows, symbol )

        if nRows < minRows:
            logger.warning( resp.text )
            logger.warning( 'Skipping %s as it has only %d rows!',
                            symbols,
                            nRows  )
            continue

        if initFlag:
            df       = tmpDf
            initFlag = False
        else:
            df = df.merge( tmpDf,
                           how = 'outer',
                           on  = [ 'Date' ] )
            
    df = df[ [ 'Date' ] + symbols ]
    df[ 'Date' ] = df[ 'Date' ].apply( pd.to_datetime )
    
    df = df.sort_values( [ 'Date' ], ascending = [ True ] )    
    df = df.interpolate( method = 'linear' )
    df = df.dropna()
    df = df.reset_index( drop = True )
    
    logger.info( 'Getting %d symbols took %0.2f seconds!',
                 len( symbols ), 
                 time.time() - t0 )
    
    return df

# ***********************************************************************
# getETFMadMeanKibot(): Get MAD and mean return of an ETF
# ***********************************************************************

def getETFMadMeanKibot( symbol,
                        nDays    = 10,
                        interval = 1,
                        maxTries = 10,
                        timeout  = 60,
                        minRows  = 100,
                        logger   = None   ):

    df = getKibotData( etfs     = [ symbol ],
                       nDays    = nDays,
                       interval = interval,
                       maxTries = maxTries,
                       timeout  = timeout,
                       minRows  = minRows,
                       logger   = logger   )
    
    retDf = pd.DataFrame( { symbol: np.log( df[ symbol ] ).pct_change().dropna() } )
    mad   = ( retDf - retDf.mean() ).abs().mean()
    mean  = retDf.mean()
    mad   = float( mad )
    mean  = float( mean )
    
    return mad, mean

# ***********************************************************************
# selectETFs(): Select ETFs
# ***********************************************************************

def sortETFs( etfList,
              nDays,
              criterion = 'ratio',
              minRows   = 10,
              logger    = None    ):

    if logger is None:
        logger = getLogger( None, 1 )

    if criterion not in [ 'mean', 'mad', 'ratio' ]:
        logger.error( 'The criterion is mean, mad, or ratio! Got %s instead!',
                      criterion )

    madList   = []
    meanList  = []
    assetList = []
    
    for symbol in etfList:

        try:
            mad, mean = getETFMadMeanKibot( symbol,
                                            nDays    = nDays,
                                            interval = 1,
                                            maxTries = 10,
                                            timeout  = 60,
                                            minRows  = minRows,
                                            logger   = logger   )
        except Exception as e:
            logger.warning( e )
            logger.warning( 'Skipping %s as could not get data!', symbol )

        assetList.append( symbol )
        madList.append( mad )
        meanList.append( mean )

    eDf = pd.DataFrame( { 'asset' : assetList,
                          'mad'   : madList,
                          'mean'  : meanList } )
    
    eDf[ 'ratio' ] = eDf[ 'mad' ] / eDf[ 'mean' ]

    eDf.sort_values( criterion, ascending = False, inplace = True )

    return eDf
