# ***********************************************************************
# Import libraries
# ***********************************************************************

import os
import sys
import pandas as pd

sys.path.append( '../' )

import utl.utils as utl
from dat.assets import ETF_HASH

# ***********************************************************************
# Input
# ***********************************************************************

etfFile = 'data/All_ETFs_Intraday.txt'
outFile = 'analysis-results/etf_avg_volumes.csv'   

# ***********************************************************************
# Analyze
# ***********************************************************************

#eDf     = pd.read_csv( etfFile, delimiter = '\t' )
symbols    = []
invSymbols = []
volumes    = []
invVolumes = []

for symbol in ETF_HASH.keys():

    invSymbol = ETF_HASH[ symbol ]
    
    etfs  = list( set( [ symbol, invSymbol, 'SPY' ] ) )
    tmpDf = utl.getKibotData( etfs        = etfs,
                              nDays       = 2000,
                              interpolate = False,
                              output      = 'volume'  )

    tmpDf = tmpDf.fillna( 0.0 )

    symbols.append( symbol )
    invSymbols.append( invSymbol )
    volumes.append( float( tmpDf[ symbol ].mean() ) )
    invVolumes.append( float( tmpDf[ invSymbol ].mean() ) )
        
outDf = pd.DataFrame( { 'symbol'         : symbols,
                        'inv_symbols'    : invSymbols,
                        'avg_volume'     : volumes,
                        'avg_inv_volume' : invVolumes } )

outDf.to_csv( outFile, index = False )

print( outDf )

