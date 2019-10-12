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

trendFlag   = True
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
# Some trend plots
# ***********************************************************************

if False:
    for varName in varNames:
        df   = getPiDf( piDir, [ varName ] )
        vec  = np.array( df[ varName ] )
        nVec = len( vec )
        t    = np.linspace( 0, nVec - 1 , nVec )
        X    = t.reshape( ( nVec, 1 ) )

        linReg = LinearRegression( fit_intercept = True )
        linReg.fit( X, vec )
        trend  = linReg.predict( X ) 

        plt.plot( t, vec, 'b', t, trend, 'r' )
        plt.xlabel( 'Minutes' )
        plt.ylabel( varName )
        plt.show()

# ***********************************************************************
# Plot correlations vs. time
# ***********************************************************************

begId    = 0
nVars    = len( varNames )
varList1 = []
varList2 = []
scores   = []

for i in range( nVars ):
    for j in range( i + 1, nVars ):

        tmpList = [ varNames[i], varNames[j] ]
        df      = getPiDf( piDir, tmpList )
        df      = df.dropna()
        nTimes  = df.shape[0]
        xVals   = []
        yVals   = []
        for tsId in range( begId, nTimes, leap ):

            vecA = np.array( df[ tmpList[0] ] )[:tsId+1]
            vecB = np.array( df[ tmpList[1] ] )[:tsId+1]
            date = np.array( df[ 'Date' ] )[tsId]

            if pd.to_datetime( date ) < minDate:
                continue

            nVec   = len( vecA )
            t      = np.linspace( 0, nVec - 1 , nVec )
            X      = t.reshape( ( nVec, 1 ) )

            linReg = LinearRegression( fit_intercept = True )
            linReg.fit( X, vecA )
            trendA = linReg.predict( X ) 

            linReg = LinearRegression( fit_intercept = True )
            linReg.fit( X, vecB )
            trendB = linReg.predict( X ) 

            vecA = vecA - trendA
            vecB = vecB - trendB
            fct  = np.linalg.norm( vecA ) * np.linalg.norm( vecB )

            if fct > 0:
                fct = 1.0 / fct

            xVals.append( tsId )
            yVals.append( fct * abs( np.dot( vecA, vecB ) ) )

        xVals  = np.array( xVals )
        yVals  = np.array( yVals )
        X      = xVals.reshape( ( len( xVals ), 1 ) )
        linReg = LinearRegression( fit_intercept = True )

        linReg.fit( X, yVals )

        score  = linReg.score( X, yVals )
        prds   = linReg.predict( X ) 

        varList1.append( varNames[i] )
        varList2.append( varNames[j] )
        scores.append( score )
        
        print( varNames[i], varNames[j], score )

        if False:
            plt.plot( xVals, yVals, 'b', xVals, prds, 'r' )
            plt.xlabel( 'Minutes' )
            plt.ylabel( tmpList[0] + '_' + tmpList[1] )
            plt.title( 'R2 = ' + str( round( score, 2 ) ) )
            plt.show()

outDf = pd.DataFrame( { 'Var1'  : varList1,
                        'Var2'  : varList2,
                        'score' : scores   } )

outDf.to_csv( 'lin_scores.csv' )

sumDf = pd.DataFrame( outDf.groupby( ['Var1'] )[ 'score' ].mean())
sumDf = sumDf.sort_values( [ 'score' ], ascending = [ False ] )

sumDf[ 'Var1' ] = sumDf.index

print( sumDf )

