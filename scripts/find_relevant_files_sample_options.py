# ***********************************************************************
# Impoort libs
# ***********************************************************************

import os
import sys
import shutil
import zipfile
import pandas as pd

sys.path.append( os.path.abspath( '../' ) )

from dat.assets import ETF_HASH, SUB_ETF_HASH, NEW_ETF_HASH, POP_ETF_HASH
from dat.assets import OPTION_ETFS, PI_ETFS
from dat.assets import INDEXES, PI_INDEXES
from dat.assets import FUTURES

# ***********************************************************************
# Set some parameters 
# ***********************************************************************

ETFS = list( ETF_HASH.keys() ) + list( ETF_HASH.values() ) +\
       list( SUB_ETF_HASH.keys() ) + list( SUB_ETF_HASH.values() ) +\
       list( NEW_ETF_HASH.keys() ) + list( NEW_ETF_HASH.values() ) +\
       list( POP_ETF_HASH.keys() ) + list( POP_ETF_HASH.values() )
ETFS = list( set( ETFS ) )

INDEXES = INDEXES + PI_INDEXES
INDEXES = list( set( INDEXES ) )

assets = ETFS + INDEXES

inDir  = 'sample_options_data'
outDir = 'sample_options_data/tracked_symbols'

# ***********************************************************************
# Get relevant samples
# ***********************************************************************

df = pd.DataFrame()
    
for item in os.listdir( inDir ):

    try:
        if item.split( '.' )[1] != 'zip':
            continue
    except:
        continue

    print( 'Processing %s...' % item )

    filePath = os.path.join( inDir, item )
    
    dirName = item.split( '.' )[0]

    csvDir = os.path.join( inDir, dirName )
    
    with zipfile.ZipFile( filePath, 'r' ) as fHd:
        fHd.extractall( csvDir )

    print( 'Extracted %s to %s..' % ( item, csvDir ) )
    
    for csvItem in os.listdir( csvDir ):

        if csvItem.split( '_' )[0] == 'L3':
            tmpDf = pd.read_csv( os.path.join( csvDir, csvItem ),
                                 usecols = [ 0, 1, 3, 4, 5, 6, 7, 8 ],
                                 names   = [ 'UnderlyingSymbol',
                                             'UnderlyingPrice',
                                             'OptionSymbol',
                                             'Type',
                                             'Expiration',
                                             'DataDate',
                                             'Strike',
                                             'Last'    ]   )
        elif csvItem.split( '_' )[0] == 'L2' and \
             csvItem.split( '_' )[1] == 'options':
                tmpDf = pd.read_csv( os.path.join( csvDir, csvItem ) )
                tmpDf = tmpDf[ [ 'UnderlyingSymbol',
                                 'UnderlyingPrice',
                                 'OptionSymbol',
                                 'Type',
                                 'Expiration',
                                 'DataDate',
                                 'Strike',
                                 'Last' ] ]
        elif csvItem.split( '_' )[0] == 'bb':
                tmpDf = pd.read_csv( os.path.join( csvDir, csvItem ) )
                tmpDf = tmpDf[ [ 'UnderlyingSymbol',
                                 'UnderlyingPrice',
                                 'OptionRoot',
                                 'Type',
                                 'Expiration',
                                 'DataDate',
                                 'Strike',
                                 'Last' ] ]
                tmpDf = tmpDf.rename( columns = { 'OptionRoot': 'OptionSymbol' } )
        else:
            continue

        symbol = list( tmpDf.UnderlyingSymbol )[0]

        if symbol not in assets:
            continue
        
        df = pd.concat( [ df, tmpDf ] )

    shutil.rmtree( csvDir )
    
df.to_pickle( os.path.join( outDir, 'relevant_option_samples.pkl' ),
              protocol = 4 ) 
