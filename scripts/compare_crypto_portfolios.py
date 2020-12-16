# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import datetime
import ast
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append( '../' )

import utl.utils as utl

# ***********************************************************************
# Input
# ***********************************************************************

actFlag = False
baseFlag = True

prtFiles = [
    'portfolios/crypto_9PM_no_zcash_no_short.json',
    'portfolios/crypto_9PM_no_zcash_ptc_no_short.json',
    'portfolios/crypto_9PM_no_zcash_ptc_no_short_gnp_1.4_4_30.json',
    'portfolios/crypto_9PM_no_zcash_ptc_no_short_gnp_1.4_4_lsp_1.0_1_num_pers_30.json',
]

legends = [
    '9 PM; No ZCash',
    '9 PM; No ZCash + PTC',
    '9 PM; No ZCash + PTC + GNP 1.4, 4, 30',
        '9 PM; No ZCash + PTC + GNP/LSP 1.4/1.0, 4/1, 30',
]

dfFile      = 'data/dfFile_crypto.pkl'
initTotVal  = 20000.0

actFile     = None
outFile     = 'analysis-results/compare_crypto.csv'

# ***********************************************************************
# Get actual data id applicable
# ***********************************************************************

if actFlag:
    actDf = pd.read_csv( actFile )
    actDf.Date = actDf.Date.apply( pd.to_datetime )
    actDf[ 'Account value' ] = actDf[ 'Account value' ].apply(
        lambda x : float(x.replace(',',''))
        )
    
# ***********************************************************************
# Get min and max dates
# ***********************************************************************

minDates = []
maxDates = []
for prtFile in prtFiles:

    if prtFile.split( '.' )[-1] == 'json':
        prtWtsHash = json.load( open( prtFile, 'r' ) )
    else:
        prtWtsHash = ast.literal_eval( open( prtFile, 'r' ).read() )
    
    minDates.append( min( prtWtsHash.keys() ) )
    maxDates.append( max( prtWtsHash.keys() ) )

minDate = max( minDates ) 
maxDate = min( maxDates ) 
#minDate = '2020-05-01'
#maxDate = '2020-07-31'

if actFlag:
    actDf = actDf[ ( actDf.Date >= minDate ) & ( actDf.Date <= maxDate ) ]
    initTotVal = list(actDf[ 'Account value' ])[0]

if baseFlag:    
    baseHash = {}
    for date in prtWtsHash:
        baseHash[ date ] = { 'BTC' : 0.3333, 'ETH': 0.3333, 'LTC': 0.3333 }
    
    json.dump( baseHash, open( 'portfolios/crypto_base.json', 'w' ) )

    prtFiles.append( 'portfolios/crypto_base.json' )
    legends.append( 'Base' )

# ***********************************************************************
# Read portfolios and plot
# ***********************************************************************

meanList  = []
stdList   = []
ratioList = []
for prtFile in prtFiles:
    prtWtsHash = ast.literal_eval( open( prtFile, 'r' ).read() )

    retDf = utl.calcBacktestReturns( prtWtsHash = prtWtsHash,
                                     dfFile     = dfFile,
                                     initTotVal = initTotVal,
                                     shortFlag  = True,
                                     invHash    = None,
                                     minDate    = minDate,
                                     maxDate    = maxDate   )

    meanList.append( retDf.Return.mean() )
    stdList.append( retDf.Return.std() ) 
    ratioList.append( retDf.Return.mean() / retDf.Return.std() )

    plt.plot( retDf.Date, retDf.EndVal )
    
if actFlag:
    plt.plot( actDf.Date, actDf[ 'Account value' ] )
    legends.append( 'Actual' )
    
plt.xlabel( 'Date' )
plt.ylabel( 'Value ($)' )
plt.legend( legends )
plt.title( 'Comparison of portfolios' )
plt.show()

compDf = pd.DataFrame( { 'prtFile'  : prtFiles,
                         'mean'     : meanList,
                         'std dev.' : stdList,
                         'ratio'    : ratioList } )
print( compDf )

compDf.to_csv( outFile, index = False )

# ***********************************************************************
# Plot return vs time of day
# ***********************************************************************
