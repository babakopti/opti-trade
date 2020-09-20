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

prtFiles = [
    'portfolios/subset_minute_mad_mean_sorted_ETFs_portfolio_60_eval_days.txt',
    'portfolios/prt_20182019_pc.json',
]

legends = [
    'sorted_ETFs_portfolio_60_eval_days',
    'Peak only classifier; max VIX 40.0',
]

dfFile      = 'data/dfFile_2020.pkl'
initTotVal  = 20000.0

outFile     = 'analysis-results/compare_ptc.csv'

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

    plt.plot( retDf.Date, retDf.EndVal )

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
