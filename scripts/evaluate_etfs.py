# ***********************************************************************
# Import libraries
# ***********************************************************************

import os, sys, dill
import random
import logging
import requests
import numpy as np
import pandas as pd

from io import StringIO

sys.path.append( os.path.abspath( '../' ) )

import utl.utils as utl

# ***********************************************************************
# Some definitions
# ***********************************************************************

ETF_LIST  = 'data/All_ETFs_Intraday.txt'
MIN_DAYS  = 365   
OUT_FILE  = 'mad_comparison.csv'

# ***********************************************************************
# Utility functions
# ***********************************************************************

def getTrendMAD( df, symbol ):

    retDf = pd.DataFrame( { symbol: np.log( df[ symbol ] ).pct_change().dropna() } )
    
    mad  = ( retDf - retDf.mean() ).abs().mean()
    mad  = float( mad )

    mean = retDf.mean()#df[ symbol ].pct_change().dropna().mean()
    mean = float( mean )
    
    return mad, mean

# ***********************************************************************
# Evaluate
# ***********************************************************************

eDf   = pd.read_csv( ETF_LIST, delimiter = '\t' )
ETFs  = list( eDf.Symbol )
descs = list( eDf.Description )

etfList  = []
madList  = []
meanList  = []
descList = []
dayList  = []

logger = utl.getLogger( None, 1 )

for i in range( len( ETFs ) ):

    symbol = ETFs[i]
    
    logger.info( 'Reading symbol %s...' % symbol )
    
    authFlag = utl.getKibotAuth()

    assert authFlag, 'Authorization not successful!'

    url  = 'http://api.kibot.com/?action=history'
    url  = url + '&symbol=%s&interval=daily&period=2000&type=ETFs&regularsession=0' \
        % symbol

    resp = requests.get( url, timeout = 20 )

    cols = [ 'Date',
             'Open',
             'High',
             'Low',
             'Close',
             'Volume' ]
    
    df    = pd.read_csv( StringIO( resp.text ), names = cols )
    df    = df.rename( columns = { 'Open' : symbol } )
    df    = df[ [ 'Date', symbol ] ]
    nRows = df.shape[0]
        
    logger.info( 'Got %d rows for %s!', nRows, symbol )

    if nRows < MIN_DAYS:
        logger.warning( 'Skipping %s: too few data, only %d days!',
                        symbol,
                        nRows )
        continue

    etfList.append( symbol )
    
    descList.append( descs[i] )
    
    mad, mean  = getTrendMAD( df, symbol )

    madList.append( mad )
    
    meanList.append( mean )

    dayList.append( nRows )

outDf = pd.DataFrame( { 'Asset': etfList,
                        'MAD'  : madList,
                        'MEAN' : meanList,
                        'Desc' : descList,
                        'Days' : dayList  } )

outDf = outDf.sort_values( [ 'MAD', 'MEAN' ],
                           ascending = [ True, False ] )

outDf.to_csv( OUT_FILE, index = False )

print( outDf.head( 30 ) )
