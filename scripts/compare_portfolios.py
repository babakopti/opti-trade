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

from dat.assets import ETF_HASH

# ***********************************************************************
# Input
# ***********************************************************************

actFlag = True

prtFiles = [
    'portfolios/nTrnDays_360_two_hours.json',
    'portfolios/nTrnDays_360_two_hours_ptc.json',
    'portfolios/nTrnDays_360_two_hours_ptc_std_coef_1.5_pers_off_4.json',
    'portfolios/actual_wt_hash_Nov6_2020.json',    
]

legends = [
    'every 2 hour',
    'every 2 hour PTC',
    'PTC + GNP 1.5, 4',
    'Actual from portfolio',    
]

dfFile      = 'data/dfFile_2020.pkl'
initTotVal  = 20000.0

actFile     = 'data/td_ametritade_balances_Nov6.csv'
outFile     = 'analysis-results/compare_nTrnDays_GNP.csv'

ETF_HASH[ 'UJB' ] = 'SJB'

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
minDate = '2020-10-10'

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
                                     shortFlag  = False,
                                     invHash    = ETF_HASH,
                                     minDate    = minDate,
                                     maxDate    = maxDate   )

    meanList.append( retDf.Return.mean() )
    stdList.append( retDf.Return.std() ) 
    ratioList.append( retDf.Return.mean() / retDf.Return.std() )

    plt.plot( retDf.Date, retDf.BegVal )

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
