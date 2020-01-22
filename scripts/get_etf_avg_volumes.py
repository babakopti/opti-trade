# ***********************************************************************
# Import libraries
# ***********************************************************************

import os
import sys
import pandas as pd

sys.path.append( '../' )

import utl.utils as utl

# ***********************************************************************
# Input
# ***********************************************************************

etfFile = 'data/All_ETFs_Intraday.txt'
outFile = 'analysis-results/etf_avg_volumes.csv'   

# ***********************************************************************
# Analyze
# ***********************************************************************

eDf     = pd.read_csv( etfFile, delimiter = '\t' )
symbols = []
volumes = []

for symbol in list( set( eDf.Symbol ) ):

    try:
        etfs  = list( set( [ symbol, 'SPY' ] ) )
        tmpDf = utl.getKibotData( etfs        = etfs,
                                  nDays       = 2000,
                                  interpolate = False,
                                  output      = 'volume'  )
        tmpDf = tmpDf.fillna( 0.0 )

        symbols.append( symbol )
        volumes.append( float( tmpDf[ symbol ].mean() ) )
        
    except Exception as e:
        print( e )
        print( 'Skipping %s...' % symbol )
        continue
        
outDf = pd.DataFrame( { 'symbol' : symbols,
                        'avg_volume' : volumes } )

outDf.to_csv( outFile, index = False )
