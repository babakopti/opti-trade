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
    'portfolios/nTrnDays_360_ptc.json',
    'portfolios/nTrnDays_30_ptc.json', 
    'portfolios/nTrnDays_60_ptc.json',
    'portfolios/nTrnDays_720_ptc.json',
    'portfolios/nTrnDays_1080_ptc.json',    
]

legends = [
    'nTrnDays 360 PTC',
    'nTrnDays 30 PTC',
    'nTrnDays 60 PTC',
    'nTrnDays 720 PTC',
    'nTrnDays 1080 PTC',
]

dfFile      = 'data/dfFile_2020.pkl'
initTotVal  = 20000.0

outFile     = 'analysis-results/compare_nTrnDays.csv'

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
#minDate = '2020-04-01'
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
