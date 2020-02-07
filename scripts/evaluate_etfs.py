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

from dat.assets import OLD_ETF_HASH, SUB_ETF_HASH, NEW_ETF_HASH

# ***********************************************************************
# Some definitions
# ***********************************************************************

OUT_FILE  = 'new_etf_comparison.csv'
NUM_EVAL_DAYS = 60
ALL_ETF_FILE = 'data/All_ETFs_Intraday.txt'
MIN_CNT = 100

# ***********************************************************************
# Generate description hash
# ***********************************************************************

eDf      = pd.read_csv( ALL_ETF_FILE, delimiter = '\t' )
symbols  = list( eDf.Symbol )
descs    = list( eDf.Description )
descHash = {}

for i in range( len( symbols ) ):

    symbol = symbols[i]

    descHash[ symbol ] = descs[i]
    
# ***********************************************************************
# Set ETFs
# ***********************************************************************

ETFs = []

searchStr = 'Palladium'

for symbol in symbols:
    if searchStr.lower() in descHash[ symbol ].lower():
        ETFs.append( symbol )

ETFs = list( set( SUB_ETF_HASH.values() ) )

# ***********************************************************************
# Utility functions
# ***********************************************************************

def getMeanMad( df, symbol ):

    tmpDf = df.copy()
    
    tmpFunc     = lambda x : pd.to_datetime( x ).date()
    tmpDf[ 'tmp' ] = tmpDf.Date.apply( tmpFunc )
    
    tmpDf = tmpDf.groupby( [ 'tmp' ], as_index = False )[ [symbol] ].mean()
    tmpDf = tmpDf.rename( columns = { 'tmp' : 'Date' } )

    tmpVec = np.log( tmpDf[ symbol ] ).pct_change().dropna() 
    
    mad  = ( tmpVec - tmpVec.mean() ).abs().mean()
    mad  = float( mad )
    mean = tmpVec.mean()
    mean = float( mean )
    
    return mad, mean

# ***********************************************************************
# Evaluate
# ***********************************************************************

etfList   = []
descList  = []
cntList   = []
madList   = []
meanList  = []

logger = utl.getLogger( None, 1 )

for i in range( len( ETFs ) ):

    symbol = ETFs[i]
    
    try:
        df = utl.getKibotData( etfs  = [ symbol ],
                               nDays = NUM_EVAL_DAYS )
    except Exception as e:
        logger.warning( e )
        continue

    etfList.append( symbol )
    descList.append( descHash[ symbol ] )
    cntList.append( df.shape[0] )
    
    mad, mean  = getMeanMad( df, symbol )

    madList.append( mad )
    meanList.append( mean )

outDf = pd.DataFrame( { 'asset': etfList,
                        'mad'  : madList,
                        'mean' : meanList,
                        'desc' : descList,
                        'cnt'  : cntList  } )

outDf[ 'abs_sharpe' ] = abs( outDf[ 'mean' ] ) / outDf[ 'mad' ]

outDf = outDf.sort_values( 'abs_sharpe',
                           ascending = False )

outDf = outDf[ outDf.cnt >= MIN_CNT ]
outDf = outDf.reset_index( drop = True )

outDf.to_csv( OUT_FILE, index = False )

print( outDf.head( 30 ) )
