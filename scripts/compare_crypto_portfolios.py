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

prtFiles = [
    'portfolios/crypto_24_hours_no_short_3AM.json',
    'portfolios/crypto_24_hours_no_short_8AM.json',    
    'portfolios/crypto_24_hours_no_short_11AM.json',
    'portfolios/crypto_24_hours_no_short_2PM.json' ,
    'portfolios/crypto_24_hours_no_short_5:30PM.json' ,
    'portfolios/crypto_24_hours_no_short_6PM.json' ,
    'portfolios/crypto_24_hours_no_short_6:30PM.json' ,
    'portfolios/crypto_24_hours_no_short_7PM.json' ,    
    'portfolios/crypto_24_hours_no_short_7:30PM.json',
    'portfolios/crypto_24_hours_no_short_8PM.json',
    'portfolios/crypto_24_hours_no_short_8:30PM.json' ,           
    'portfolios/crypto_24_hours_no_short_9PM.json' ,
    'portfolios/crypto_24_hours_no_short_9:15PM.json' ,               
    'portfolios/crypto_24_hours_no_short_9:30PM.json' ,
    'portfolios/crypto_24_hours_no_short_9:45PM.json' ,           
    'portfolios/crypto_24_hours_no_short_10PM.json',
    'portfolios/crypto_24_hours_no_short_10:30PM.json',    
    'portfolios/crypto_24_hours_no_short_11PM.json',
    'portfolios/crypto_24_hours_no_short_11:30PM.json',    
    'portfolios/crypto_24_hours_no_short_12AM.json',
]

legends = [
    '3 AM',
    '8 AM',    
    '11 AM',
    '2 PM',
    '5:30 PM',
    '6 PM',
    '6:30 PM',
    '7 PM',
    '7:30 PM',    
    '8 PM',
    '8:30 PM',        
    '9 PM',
    '9:15 PM',
    '9:30 PM',
    '9:45 PM',
    '10 PM',
    '10:30 PM',
    '11 PM',
    '11:30',
    'Mid night',
]

dfFile      = 'data/dfFile_crypto.pkl'
initTotVal  = 20000.0

actFile     = None
outFile     = 'analysis-results/compare_crypto_prts_Jul_Oct_2020.csv'

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
minDate = '2020-07-01'
#maxDate = '2020-07-31'

if actFlag:
    actDf = actDf[ ( actDf.Date >= minDate ) & ( actDf.Date <= maxDate ) ]
    initTotVal = list(actDf[ 'Account value' ])[0]
    
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
