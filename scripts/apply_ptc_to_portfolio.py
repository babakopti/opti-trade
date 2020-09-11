# ***********************************************************************
# Import libraries
# ***********************************************************************

import os
import sys
import json
import pandas as pd

sys.path.append( '..' )

from ptc.ptc import PTClassifier

# ***********************************************************************
# Main input params
# ***********************************************************************

ptcFlag    = True
vixMrgFlag = True

prtFile    = 'portfolios/portfolio_every_3_hours_assets_5.json'
minVix     = None
maxVix     = None
datDir     = 'data'
ptcDir     = 'pt_classifiers'
outPrtFile = 'portfolios/test.json'

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

        ptcDfFile = '%s/%s_VIX_daily.pkl' % ( datDir, symbol )
        
        if vixMrgFlag:
            symDf = pd.read_pickle( '%s/%s.pkl' % ( datDir, symbol ) )
            symDf[ 'Date' ] = symDf.Date.apply( \
                            lambda x : x.strftime( '%Y-%m-%d' ) )
            symDf = symDf.groupby( 'Date', as_index = False ).mean()

            vixDf = pd.read_pickle( '%s/VIX.pkl' % datDir )
            vixDf[ 'Date' ] = vixDf.Date.apply( \
                            lambda x : x.strftime( '%Y-%m-%d' ) )
            vixDf = vixDf.groupby( 'Date', as_index = False ).mean()
            
            symDf = symDf.merge( vixDf, on = 'Date', how = 'left' )            
            symDf = symDf.interpolate( method = 'linear' )
            symDf.to_pickle( ptcDfFile )

        ptcObj = PTClassifier( symbol      = symbol,
                               dfFile      = ptcDfFile,
                               ptThreshold = 1.0e-2,
                               nAvgDays    = 7,
                               nPTAvgDays  = None,
                               testRatio   = 0.1,
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

for datStr in prtWtsHash:

    tmpHash  = prtWtsHash[ dateStr ]
    snapDate = pd.to_datetime( dateStr )
    
    for symbol in tmpHash:

        ptcDfFile = '%s/%s_VIX_daily.pkl' % ( datDir, symbol )
        
        dayDf = pd.read_pickle( ptcDfFile )

        dayDf = dayDf[ [ 'Date', symbol, 'VIX' ] ]

        dayDf[ 'Date' ] = dayDf.Date.astype( 'datetime64[ns]' )
        
        tmpDate = snapDate - pd.DateOffset( days = 60 )
        
        dayDf = dayDf[ (dayDf.Date >= tmpDate) & (dayDf.Date <= snapDate) ]

        dayDf[ 'vel' ] = np.gradient( dayDf[ symbol ], 2 )
        dayDf[ 'acl' ] = np.gradient( dayDf[ 'vel' ], 2 )

        dayDf[ 'avgAcl' ] = dayDf.acl.rolling( min_periods = 1,
                                               window = 7 ).mean()
        dayDf[ 'feature' ] = dayDf.acl - dayDf.avgAcl        
                 
        obj = pickle.load( open( '%s/ptc_%s.pkl' % ( ptcDir, symbol ), 'rb' ) )

        dayDf = dayDf.sort_values( [ 'Date' ], ascending = True )
        
        val = list( dayDf.feature )[-1]

        X = np.array( [ val ] ).reshape( ( 1, 1 ) )
        
        ptTag = obj.predict( X )[0]

        vixVal = list( dayDf.VIX )[-1]

        if minVix is not None and vixVal < minVix:
            continue

        if maxVix is not None and vixVal > maxVix:
            continue
        
        if ptTag == 1:
            tmpHash[ symbol ] = -abs( tmpHash[ symbol ] )
        # if ptTag == 2:
        #     tmpHash[ symbol ] = abs( tmpHash[ symbol ] )

    prtWtsHash[ dateStr ] = tmpHash

# ***********************************************************************
# Write the adjusted portfolio
# ***********************************************************************
    
with open( outPrtFile, 'w' ) as fp:
    json.dump( prtWtsHash, fp )        
