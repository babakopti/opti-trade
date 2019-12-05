# ***********************************************************************
# Import libraries
# ***********************************************************************

import os, sys, dill
import ast
import time
import datetime
import random
import talib
import numpy as np
import pandas as pd

# ***********************************************************************
# Set some parameters
# ***********************************************************************

inpHash     = { 'portfolios/sign_trick_20191022.txt': 'data/dfFile_2017plus.pkl',
                'portfolios/portfolio_2017.txt': 'data/dfFile_2016_2017.pkl',
                'portfolios/portfolio_2016.txt': 'data/dfFile_2014_2016.pkl',
                'portfolios/portfolio_2015.txt': 'data/dfFile_2014_2016.pkl',
                'portfolios/portfolio_2014.txt': 'data/dfFile_2011_2014.pkl',
                'portfolios/portfolio_2013.txt': 'data/dfFile_2011_2014.pkl',
                'portfolios/portfolio_crash_macd.txt': 'data/dfFile_2005_2010.pkl' }

nPrdDays    = 1
nPrdTimes   = int( nPrdDays * 19 * 60 )

sumFile     = 'trend_comparison.csv'

# ***********************************************************************
# Some utility functions
# ***********************************************************************

def getActTrends( curDate, df, assets ):

    begDate = pd.to_datetime( curDate )
    endDate = begDate + datetime.timedelta( days = nPrdDays )

    df = df[ df.Date >= begDate ]
    df = df[ df.Date < endDate ]
    df = df.sort_values( [ 'Date' ] )

    trendHash = {}

    for asset in assets:
        tmpVec = np.array( df[ asset ] )[:nPrdTimes]

        if len( tmpVec ) == 0:
            trend = 0
            print( 'Warning: no data found for', curDate, asset )
        else:
            trend = np.mean( tmpVec ) - tmpVec[0]

        trendHash[ asset ] = trend

    return trendHash

# ***********************************************************************
# Evaluate
# ***********************************************************************

dateList     = []
assetList    = []
actTrendList = []
prdTrendList = []
successList  = []

for prtFile in inpHash:

    t0 = time.time()
    
    print( 'Processing portfolio', prtFile, '...' )
    
    tmpStr = open( prtFile, 'r' ).read()
    wtHash = ast.literal_eval( tmpStr )
    dfFile = inpHash[ prtFile ]
    tmpDf  = pd.read_pickle( dfFile )
    
    for curDate in wtHash:
        prdTrendHash = wtHash[ curDate ]
        assets       = prdTrendHash.keys()
        actTrendHash = getActTrends( curDate,
                                     tmpDf,
                                     assets    )
        assets = actTrendHash.keys()

        if len( assets ) == 0:
            continue
        
        for asset in assets:
            dateList.append( curDate )
            assetList.append( asset )

            tmp = prdTrendHash[ asset ] * actTrendHash[ asset ]
            
            if tmp > 0:
                success = 1
            else:
                success = 0

            successList.append( success )

    print( 'Processing took %0.2f seconds!' % ( time.time() - t0 ) )
                
outDf = pd.DataFrame( { 'Date'     : dateList,
                        'Asset'    : assetList,
                        'Success'  : successList      }   )

print( '1-Mean success rate:', outDf.Success.mean() )

sumDf = outDf.groupby( 'Date', as_index = False )[ 'Success' ].mean()

print( '2-Mean success rate:', sumDf.Success.mean() )

sumDf[ 'Year' ] = sumDf.Date.apply( lambda x : pd.to_datetime(x).year )
sumDf = sumDf.groupby( 'Year', as_index = False )[ 'Success' ].mean()

print( '3-Mean success rate:', sumDf.Success.mean() )

sumDf.to_csv( sumFile, index = False )

