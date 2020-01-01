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

def getKibotData( etfs        = [],
                  futures     = [],
                  stocks      = [],
                  indexes     = [],
                  nDays       = 365,
                  interval    = 1,
                  maxTries    = 10,
                  timeout     = 60,
                  smoothCount = 1000,
                  smoothConf  = 10,
                  minRows     = 100,
                  logger      = None   ):

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

        # Remove anomalies
        
        logger.info( 'Checking %s for anomalies...', symbol )
        
        tmpDf[ 'smooth' ] = tmpDf[ symbol ].rolling( smoothCount, 
                                                     win_type = 'blackman',
                                                     center   = True ).mean()
        tmpDf[ 'smooth' ] = tmpDf[ 'smooth' ].fillna( tmpDf[ symbol ] )
        tmpDf[ 'smooth' ] = tmpDf[ symbol ] - tmpDf[ 'smooth' ]
        
        tmpStd  = tmpDf[ 'smooth' ].std()
        tmpMean = tmpDf[ 'smooth' ].mean()
        
        tmpDf[ 'smooth' ] = ( tmpDf[ 'smooth' ] - tmpMean ).abs()
        tmpDf[ 'smooth' ] = smoothConf * tmpStd - tmpDf[ 'smooth' ]

        nAnoms = tmpDf[ tmpDf.smooth < 0 ].shape[0]
        tmpDf  = tmpDf[ tmpDf.smooth >= 0 ]
        nRows  = tmpDf.shape[0]

        if nAnoms > 0:
            logger.info( 'Removed %d anomalies for %s!', nAnoms, symbol )
        elif nAnoms == 0:
            logger.info( 'No anomalies found for %s!', symbol )
        
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

def getMadMeanKibot( symbol,
                     sType    = 'ETF',
                     nDays    = 10,
                     interval = 1,
                     maxTries = 10,
                     timeout  = 60,
                     minRows  = 100,
                     logger   = None   ):

    if sType == 'ETF':
        df = getKibotData( etfs     = [ symbol ],
                           nDays    = nDays,
                           interval = interval,
                           maxTries = maxTries,
                           timeout  = timeout,
                           minRows  = minRows,
                           logger   = logger   )
    elif sType == 'futures':
        df = getKibotData( futures  = [ symbol ],
                           nDays    = nDays,
                           interval = interval,
                           maxTries = maxTries,
                           timeout  = timeout,
                           minRows  = minRows,
                           logger   = logger   )
    
    retDf = pd.DataFrame( { symbol: np.log( df[ symbol ] ).pct_change().dropna() } )
    mean  = retDf.mean()
    mad   = ( retDf - mean ).abs().mean()
    mean  = float( mean )    
    mad   = float( mad )
    
    return mad, mean

# ***********************************************************************
# selectETFs(): Select ETFs
# ***********************************************************************

def sortAssets( symbols,
                nDays,
                sType     = 'ETF',
                minRows   = 10,
                logger    = None    ):

    if logger is None:
        logger = getLogger( None, 1 )

    madList   = []
    meanList  = []
    assetList = []
    
    for symbol in symbols:

        try:
            mad, mean = getMadMeanKibot( symbol,
                                         sType    = sType,
                                         nDays    = nDays,
                                         interval = 1,
                                         maxTries = 10,
                                         timeout  = 60,
                                         minRows  = minRows,
                                         logger   = logger   )
        except Exception as e:
            logger.warning( e )
            logger.warning( 'Skipping %s as could not get data!', symbol )
            continue

        assetList.append( symbol )
        madList.append( mad )
        meanList.append( mean )

    eDf = pd.DataFrame( { 'asset' : assetList,
                          'mad'   : madList,
                          'mean'  : meanList } )
    
    eDf[ 'score' ] = abs( eDf[ 'mean' ] ) / eDf[ 'mad' ]

    eDf.sort_values( 'score', ascending = False, inplace = True )

    eDf.reset_index( drop = True, inplace = True )
    
    return eDf

# ***********************************************************************
# calcPrtReturns: Calculate portfolio returns over a period
# ***********************************************************************

def calcPrtReturns( prtWtsHash,
                    dfFile,
                    initTotVal,
                    shortFlag = True,
                    invHash   = None,
                    minDate   = None,
                    maxDate   = None   ):

    # Get all dates in portfolio
    
    dates = list( prtWtsHash.keys() )
    dates = sorted( dates )

    # Set the assets
    
    assets = []
    for date in dates:
        assets += list( prtWtsHash[ date ].keys() )

    assets = list( set( assets ) )

    # Check inverse asset hash if applicable

    if not shortFlag:
        for asset in assets:
            assert asset in invHash.keys(), \
                '%s not found in invHash' % asset

    # Read, check and clean price data frame
    
    dataDf = pd.read_pickle( dfFile )

    for asset in assets:
        assert asset in dataDf.columns, \
            'Symbol %s not found in %s' % ( asset, dfFile )

    if not shortFlag:
        invAssets = list( invHash.values() )
        for asset in invAssets:
            assert asset in dataDf.columns, \
                'Symbol %s not found in %s' % ( asset, dfFile )
    
    dataDf = dataDf[ [ 'Date' ] + assets ]

    dataDf[ 'Date' ] = dataDf[ 'Date' ].apply( pd.to_datetime )
    
    dataDf.sort_values( 'Date', inplace = True )
    dataDf.drop_duplicates( inplace = True )
    dataDf.dropna( inplace = True )
    dataDf.reset_index( drop = True, inplace = True  )


        # Set min and max dates
    
    if minDate is None:
        minDate = max( pd.to_datetime( dates[0] ),
                       dataDf.Date.min() )
    else:
        minDate = pd.to_datetime( minDate )

    if maxDate is None:
        maxDate = min( pd.to_datetime( dates[-1] ),
                       dataDf.Date.max() )
    else:
        maxDate = pd.to_datetime( maxDate )

    # Loop through portfolio dates and calculate values
    
    nDates     = len( dates )
    begTotVal  = initTotVal
    begDates   = []
    begTotVals = []
    
    for itr in range( nDates ):

        if pd.to_datetime( dates[itr] ) < minDate:
            continue

        if pd.to_datetime( dates[itr] ) > maxDate:
            continue

        begDate = pd.to_datetime( dates[itr] )

        begDates.append( begDate )
        begTotVals.append( begTotVal )

        if itr < nDates - 1:
            endDate = pd.to_datetime( dates[itr+1] )
        elif itr == nDates - 1:
            tmp1    = pd.to_datetime( dates[nDates-1] )
            tmp2    = pd.to_datetime( dates[nDates-2] )
            nDays   = ( tmp1 - tmp2 ).days
            endDate = begDate + datetime.timedelta( days = nDays )

        tmpDf = dataDf[ dataDf.Date >= begDate ]
        tmpDf = tmpDf[ tmpDf.Date <= endDate ]
    
        if tmpDf.shape[0] == 0:
            print( 'Skipping date', date )
            continue

        tmpDf.sort_values( 'Date', inplace = True )

        wtHash = defaultdict( float )

        for asset in prtWtsHash[ dates[itr] ]:
            wtHash[ asset ] = prtWtsHash[ dates[itr] ][ asset ]

        tmp1 = 0.0
        tmp2 = 0.0
        for asset in assets:

            wt = wtHash[ asset ]

            if wt == 0:
                continue
            elif wt > 0 or shortFlag:
                begPrice = list( tmpDf[ asset ] )[0]
                endPrice = list( tmpDf[ asset ] )[-1]
                qty      = int( wt * begTotVal / begPrice )
            elif wt < 0 and not shortFlag:
                invAsset = invHash[ asset ]
                begPrice = list( tmpDf[ invAsset ] )[0]
                endPrice = list( tmpDf[ invAsset ] )[-1]
                qty      = int( abs( wt ) * begTotVal / begPrice )
        
            tmp1 += qty * begPrice
            tmp2 += qty * endPrice
        
        cash = begTotVal - tmp1

        assert cash >= 0, \
            'Cash should be non-negative! Date %s' % str( begDate )

        endTotVal = tmp2 + cash

        # Update begTotVal for next item in loop
    
        begTotVal = endTotVal

    retDf = pd.DataFrame( { 'Date'  : begDates,
                            'Value' : begTotVals } )

    retDf[ 'Return' ] = retDf[ 'Value' ].pct_change()
    retDf[ 'Return' ] = retDf[ 'Return' ].fillna( 0 )
    retDf[ 'Change' ] = ( retDf[ 'Value' ] / initTotVal ) - 1.0

    return retDf
