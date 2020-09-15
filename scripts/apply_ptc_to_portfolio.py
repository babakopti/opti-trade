# ***********************************************************************
# Import libraries
# ***********************************************************************

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

sys.path.append( '..' )

from ptc.ptc import PTClassifier

# ***********************************************************************
# Main input params
# ***********************************************************************

ptcFlag    = True
vixMrgFlag = False

prtFile    = 'portfolios/portfolio_every_3_hours_assets_5.json'
minVix     = None
maxVix     = 40
datDir     = 'data'
ptcDir     = 'pt_classifiers'
outPrtFile = 'portfolios/test_max_vix_40.json'

# ***********************************************************************
# Read original portfolio and get symbols
# ***********************************************************************

prtWtsHash = json.load( open( prtFile, 'r' ) )

symbols = []
for dateStr in prtWtsHash:
    symbols += list( prtWtsHash[ dateStr ].keys() )

symbols = list( set( symbols ) )

# ***********************************************************************
# Build PT classifiers if applicable
# ***********************************************************************

if ptcFlag:

    for symbol in symbols:

        symFile = '%s/%s.pkl' % ( datDir, symbol )
        vixFile = '%s/VIX.pkl' % datDir

        ptcObj = PTClassifier( symbol      = symbol,
                               symFile     = symFile,
                               vixFile     = vixFile,
                               ptThreshold = 1.0e-2,
                               nAvgDays    = 7,
                               nPTAvgDays  = None,
                               testRatio   = 0,
                               method      = 'bayes',
                               minVix      = minVix,
                               maxVix      = maxVix,
                               logFileName = None,                    
                               verbose     = 1          )

        ptcObj.classify()

        ptcObj.save( '%s/ptc_%s.pkl' % ( ptcDir, symbol ) )

# ***********************************************************************
# Apply ptc model
# ***********************************************************************

allDates = sorted( list( prtWtsHash.keys() ) )

peakAdjCnt = 0
peakAdjDates  = []

for dateStr in allDates:

    print( 'Adjsuting snapDate %s' % dateStr )
    
    tmpHash  = prtWtsHash[ dateStr ]
    snapDate = pd.to_datetime( dateStr )
    
    df = pd.read_pickle( 'data/dfFile_2020.pkl' )
    
    df[ 'Date' ] = df.Date.astype( 'datetime64[ns]' )
    
    tmpDate = snapDate - pd.DateOffset( days = 21 )
    
    df = df[ ( df.Date >= tmpDate ) & ( df.Date <= snapDate ) ]
    
    df[ 'Date' ] = df.Date.apply( lambda x : x.strftime( '%Y-%m-%d' ) )
    
    dayDf = df.groupby( 'Date', as_index = False ).mean()
    
    dayDf[ 'Date' ] = dayDf.Date.astype( 'datetime64[ns]' )
    
    dayDf = dayDf.sort_values( [ 'Date' ], ascending = True )

    vixVal = list( dayDf.VIX )[-1]
    
    if minVix is not None and vixVal < minVix:
        continue

    if maxVix is not None and vixVal > maxVix:
        continue
    
    for symbol in tmpHash:
        
        dayDf[ 'vel' ] = np.gradient( dayDf[ symbol ], 2 )
        dayDf[ 'acl' ] = np.gradient( dayDf[ 'vel' ], 2 )
        
        tmpDate = snapDate - pd.DateOffset( days = 7 )
        avgAcl = dayDf[ dayDf.Date >= tmpDate ].acl.mean()
        symVal = list( dayDf.acl )[-1] - avgAcl
        
        obj = pickle.load( open( '%s/ptc_%s.pkl' % ( ptcDir, symbol ), 'rb' ) )

        X = np.array( [ symVal ] ).reshape( ( 1, 1 ) )
        
        ptTag = obj.predict( X )[0]
        
        if ptTag == 1:
            print( 'Peak detected for %s at %s' % (symbol, dateStr))            
            if tmpHash[ symbol ] > 0:
                tmpHash[ symbol ] = -tmpHash[ symbol ]
                peakAdjCnt += 1
                peakAdjDates.append( dateStr )
        # if ptTag == 2:
        #     tmpHash[ symbol ] = abs( tmpHash[ symbol ] )

    prtWtsHash[ dateStr ] = tmpHash

print( 'Changed weight sign %d times' % peakAdjCnt )
print( 'Changed weights on dates: %s' % str(set(peakAdjDates)) )

# ***********************************************************************
# Write the adjusted portfolio
# ***********************************************************************
    
with open( outPrtFile, 'w' ) as fp:
    json.dump( prtWtsHash, fp )        
