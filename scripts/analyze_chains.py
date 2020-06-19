# ***********************************************************************
# Import libraries
# ***********************************************************************

import os
import sys
import dill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from datetime import datetime

sys.path.append( os.path.abspath( '../' ) )

import utl.utils as utl
from dat.assets import OPTION_ETFS as ETFS
from dat.assets import FUTURES, INDEXES

from prt.prt import MfdOptionsPrt
from brk.tdam import Tdam

# ***********************************************************************
# Some parameters
# ***********************************************************************

snapDate  = pd.to_datetime( '2020-06-19' )

modFile   = 'option_model_2020-06-18_14:31:59.dill'

cash      = 2000
maxPriceC = 0.1 * cash
maxPriceA = 0.3 * cash
minProb   = 0.48
maxMonths = 3

INDEXES  = list( set( INDEXES ) - { 'TRAN', 'TYX' } )
ETFS     = list( set( ETFS ) - { 'OIH' } )
ASSETS   = ETFS

# ***********************************************************************
# Fix the log thing
# ***********************************************************************

mfdMod = dill.load( open( modFile, 'rb' ) )

mfdMod.logger = utl.getLogger( None, 1 )
mfdMod.ecoMfd.logger = utl.getLogger( None, 1 )

mfdMod.save( modFile )

# ***********************************************************************
# Read option chain from td
# ***********************************************************************

TOKEN_FILE = '../brk/tokens/refresh_token_pany_2020-09-17.txt'

with open( TOKEN_FILE, 'r' ) as fHd:
    REFRESH_TOKEN = fHd.read()[:-1]

OPTION_ACCOUNT_ID = '868894929'

td = Tdam( refToken = REFRESH_TOKEN, accountId = OPTION_ACCOUNT_ID )
        
options = []

for symbol in ASSETS:
            
    print( 'Getting options for %s...' % symbol )
    
    tmpList = td.getOptionsChain( symbol )
    
    options += tmpList
    
    print( 'Found %d options contracts!' % len( options ) )

# ***********************************************************************
# Get asset hash
# ***********************************************************************
    
td = Tdam( refToken = REFRESH_TOKEN, accountId = OPTION_ACCOUNT_ID )
        
assetHash = {}
        
for symbol in ETFS:
    assetHash[ symbol ] = td.getQuote( symbol, 'last' )
    
for symbol in FUTURES:
    val, date = utl.getYahooLastValue( symbol,
                                       'futures' )
    assetHash[ symbol ] = val

for symbol in INDEXES:
    val, date = utl.getYahooLastValue( symbol,
                                       'index' )
    assetHash[ symbol ] = val
    
# ***********************************************************************
# Build prt object
# ***********************************************************************

minDate = snapDate + pd.DateOffset( days   = 1 )
maxDate = snapDate + pd.DateOffset( months = maxMonths )

prtObj = MfdOptionsPrt( modFile     = modFile,
                        assetHash   = assetHash,
                        curDate     = snapDate,
                        minDate     = minDate,
                        maxDate     = maxDate,
                        minProb     = minProb,
                        rfiDaily    = 0.0,
                        tradeFee    = 0.75,
                        nDayTimes   = 1140,
                        logFileName = None,                    
                        verbose     = 1          )
    
# ***********************************************************************
# Select options and put in data frame format
# ***********************************************************************

if False:
    tmpHash = defaultdict(list)
    for option in options:
        option[ 'Prob' ] = prtObj.getProb( option )
        for col in option:
            tmpHash[ col ].append( option[col] )
        pd.DataFrame( tmpHash ).to_pickle( 'pre_filt_options.pkl' )

options = prtObj.filterOptions( options, maxPriceC )

print( 'Options count after filtering:', len( options ) )

if len( options ) == 0:
    print( 'No options found!' )
    sys.exit()

optHash = defaultdict(list)

for option in options:
    option[ 'Prob' ] = prtObj.getProb( option )
    for col in option:
        optHash[ col ].append( option[col] )

optDf = pd.DataFrame( optHash )

optDf = optDf[ optDf.Prob >= minProb ]

print( 'Call options:', optDf[ optDf[ 'type' ] == 'call' ].shape[0] )
print( 'Put options:', optDf[ optDf[ 'type' ] == 'put' ].shape[0] )


