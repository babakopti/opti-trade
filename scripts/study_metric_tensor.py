# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from utils import getPiDf

# ***********************************************************************
# Set some parameters and read data
# ***********************************************************************

minDate     = pd.to_datetime( '2012-01-01' )
leap        = 5000

piDir       = '/Users/babak/workarea/data/pitrading_data'
matFile     = 'mat.npy'

indices     = [ 'INDU', 'NDX', 'SPX', 'COMPX', 'RUT',  'OEX',  
                'MID',  'SOX', 'RUI', 'RUA',   'TRAN', 'HGX',  
                'TYX',  'HUI', 'XAU'                       ] 

futures     = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

ETFs        = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH', 
                'SMH', 'XLE', 'XLF', 'XLU', 'EWJ'          ]

stocks      = [ 'MMM',  'AXP', 'AAPL', 'BA', 'CAT',  'CVX',
                'CSCO', 'KO',  'XOM',  'GS',  'HD',  'INTC',
                'IBM', 'JNJ',  'JPM',  'MCD', 'MRK', 'MSFT', 
                'NKE', 'PFE',  'PG',   'TRV', 'UTX', 'UNH', 
                'VZ',  'WMT',  'WBA', 'DIS'                ]
forex       = [ 'USDJPY', 'USDCHF', 'USDCAD', 'NZDUSD',
                'GBPUSD', 'EURUSD', 'AUDUSD'               ]

varNames    = ETFs + indices

# ***********************************************************************
# Loop through vars and plot metric over time
# ***********************************************************************

begId    = 0
nVars    = len( varNames )
varList1 = []
varList2 = []
sScores  = []
mScores  = []

for i in range( nVars ):
    for j in range( i + 1, nVars ):

        tmpList = [ varNames[i], varNames[j] ]
        df      = getPiDf( piDir, tmpList )
        df      = df.dropna()
 
        nTimes  = df.shape[0]
        nLeaps  = 0
        dates   = []
        sVals   = []
        mVals   = []
        yVals   = []
        for tsId in range( begId, nTimes, leap ):

            nLeaps += 1

            vecA = np.array( df[ tmpList[0] ] )[:tsId+1]
            vecB = np.array( df[ tmpList[1] ] )[:tsId+1]
            date = np.array( df[ 'Date' ] )[tsId]

            if pd.to_datetime( date ) < minDate:
                continue

            fct  = np.linalg.norm( vecA ) * np.linalg.norm( vecB )

            if fct > 0:
                fct = 1.0 / fct

            dates.append( date )
            sVals.append( np.sum( vecA ) + np.sum( vecB ) )
            mVals.append( np.sum( vecA ) * np.sum( vecB ) )
            yVals.append( fct * abs( np.dot( vecA, vecB ) ) )

        X = np.array( sVals )
        X = X.reshape( ( len( sVals ), 1 ) )
        y = np.array( yVals )
        linReg = LinearRegression( fit_intercept = True )
        linReg.fit( X, y )

        sScore = linReg.score( X, y )

        X = np.array( mVals )
        X = X.reshape( ( len( mVals ), 1 ) )
        y = np.array( yVals )
        linReg = LinearRegression( fit_intercept = True )
        linReg.fit( X, y )

        mScore = linReg.score( X, y )

        varList1.append( varNames[i] )
        varList2.append( varNames[j] )
        sScores.append( sScore )
        mScores.append( mScore )

        print( sScore, mScore )

outDf = pd.DataFrame( { 'Var1'   : varList1,
                        'Var2'   : varList2,
                        'sScore' : sScores,
                        'mScore' : mScores } )
        
print( outDf.head() )

outDf.to_csv( 'lin_scores.csv' )

sumDf = pd.DataFrame(df.groupby( ['Var1'] )[ 'sScore' ].mean())
sumDf = sumDf.sort_values( [ 'sScore' ], ascending = [ False ] )

sumDf[ 'Var1' ] = sumDf.index

tmpHash = {}

for i in range( sumDf.shape[0] ):
    if sumDf.Var1[i] in ETFs:
        tmpHash[sumDf.Var1[i]] = sumDf.sScore[i]

print( tmpHash )

