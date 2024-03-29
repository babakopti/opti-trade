# ***********************************************************************
# Import libraries
# ***********************************************************************

import os
import sys
import ast
import json
import pickle
import numpy as np
import pandas as pd

sys.path.append( '..' )

import ptc.ptc as ptc

# ***********************************************************************
# Main input params
# ***********************************************************************

ptcFlag    = True
prtFile    = 'portfolios/crypto_9PM_raw_no_zcash.json'
minVix     = None
maxVix     = 60
datDir     = 'data'
ptcDir     = 'pt_classifiers'
outPrtFile = 'portfolios/crypto_9PM_raw_no_zcash_ptc.json'

# ***********************************************************************
# Read original portfolio and get symbols
# ***********************************************************************

if prtFile.split( '.' )[-1] == 'json':
    prtWtsHash = json.load( open( prtFile, 'r' ) )
else:
    prtWtsHash = ast.literal_eval( open( prtFile, 'r' ).read() )

symbols = []
for dateStr in prtWtsHash:
    symbols += list( prtWtsHash[ dateStr ].keys() )

symbols = list( set( symbols ) )

# ***********************************************************************
# Some utility defines
# ***********************************************************************

def buildPTC( symList ):

    for symbol in symList:

        vixFile = os.path.join( 'data',
                                'VIX.pkl' )
        
        symFile = os.path.join( 'data',
                                '%s.pkl' % symbol )

        ptcObj  = ptc.PTClassifier( symbol      = symbol,
                                    symFile     = symFile,
                                    vixFile     = vixFile,
                                    ptThreshold = 1.0e-2,
                                    nPTAvgDays  = None,
                                    testRatio   = 0,
                                    method      = 'bayes',
                                    minVix      = minVix,
                                    maxVix      = maxVix,
                                    minTrnDate  = '2019-01-01',
                                    logFileName = None,                    
                                    verbose     = 1          )

        ptcObj.classify()

        ptcFile = os.path.join( ptcDir,
                                'ptc_' + symbol + '.pkl' )
            
        print( 'Saving the classifier to %s' % ptcFile )
            
        ptcObj.save( ptcFile )

def adjustPTC( wtHash, snapDate ):

    dayDf = pd.read_pickle( 'data/dfFile_crypto.pkl' )        

    dayDf[ 'Date' ] = dayDf.Date.astype( 'datetime64[ns]' )
    
    minDate = snapDate - \
        pd.DateOffset( days = 7 )
    
    dayDf = dayDf[ ( dayDf.Date >= minDate ) &
                   ( dayDf.Date <= snapDate ) ]
    
    dayDf[ 'Date' ] = dayDf.Date.\
        apply( lambda x : x.strftime( '%Y-%m-%d' ) )
    
    dayDf = dayDf.groupby( 'Date', as_index = False ).mean()
    
    dayDf[ 'Date' ] = dayDf.Date.astype( 'datetime64[ns]' )
        
    dayDf = dayDf.sort_values( [ 'Date' ], ascending = True )

    vixVal = list( dayDf.VIX )[-1]

    if minVix is not None and vixVal < minVix:
        return wtHash

    if maxVix is not None and vixVal > maxVix:
        return wtHash

    for symbol in wtHash:

        dayDf[ 'vel' ] = np.gradient( dayDf[ symbol ], 2 )
        dayDf[ 'acl' ] = np.gradient( dayDf[ 'vel' ], 2 )
        fea1 = list( dayDf.acl )[-1]
        ptcFile = os.path.join( ptcDir,
                                'ptc_' + symbol + '.pkl' )
        obj = pickle.load( open( ptcFile, 'rb' ) )
        X = np.array( [ [ fea1 ] ] )
        ptTag = obj.predict( X )[0]
        prob  = max( obj.predict_proba( X )[0] )
        
        if ptTag == ptc.PEAK:
            print( 'A peak is detected for %s' % symbol )
            wtHash[ symbol ] = -abs( wtHash[ symbol ] )
        if ptTag == ptc.TROUGH:
            print( 'A trough is detected for %s' % symbol )
            wtHash[ symbol ] = abs( wtHash[ symbol ] )
            
    # Re-normalize 
    sumAbs = sum( [abs(x) for x in wtHash.values()] )
    sumAbsInv = 1.0
    if sumAbs > 0:
        sumAbsInv = 1.0 / sumAbs

    for symbol in wtHash:
        wtHash[ symbol ] = sumAbsInv * wtHash[ symbol ]
        
    return wtHash

# ***********************************************************************
# Build PT classifiers if applicable
# ***********************************************************************

if ptcFlag:
    buildPTC( symbols )

# ***********************************************************************
# Apply ptc model
# ***********************************************************************

allDates = sorted( list( prtWtsHash.keys() ) )

for dateStr in allDates:

    print( 'Adjsuting snapDate %s' % dateStr )
    
    tmpHash  = prtWtsHash[ dateStr ]
    snapDate = pd.to_datetime( dateStr )
    tmpHash  = adjustPTC( tmpHash, snapDate )

    prtWtsHash[ dateStr ] = tmpHash

# ***********************************************************************
# Write the adjusted portfolio
# ***********************************************************************
    
with open( outPrtFile, 'w' ) as fp:
    json.dump( prtWtsHash, fp )        
