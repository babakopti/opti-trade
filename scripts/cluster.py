# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering

from utils import getPiDf

# ***********************************************************************
# Set some parameters and read data
# ***********************************************************************

nClusters   = 3
matFlag     = True

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

varNames    = ETFs

# ***********************************************************************
# Remove cached matrix if flag is on
# ***********************************************************************

if matFlag:
    if os.path.exists( matFile ):
        os.remove( matFile )

# ***********************************************************************
# Define metric function
# ***********************************************************************

def metric( X ):

    nItems = len( X )

    try:
        mat = np.load( matFile )
    except:
        mat = np.empty( shape = ( nItems, nItems ), dtype = 'd' )
        mat.fill( -1.0 )
        pass

    for i in range( nItems ):
        for j in range( nItems ):

            if mat[i,j] >= 0:
                continue

            if i == j:
                mat[i,j] = 0.0
                continue
            
            varNames = [ X[i][0], X[j][0] ]
            df       = getPiDf( piDir, varNames )
            df       = df.dropna()
            
            vecA     = np.array( df[ varNames[0] ] )
            vecB     = np.array( df[ varNames[1] ] )
            fct      = np.linalg.norm( vecA ) * np.linalg.norm( vecB )

            if fct > 0:
                fct = 1.0 / fct

            mat[i,j] = 1.0 - fct * abs( np.dot( vecA, vecB ) )
            mat[j,i] = mat[i,j]

    np.save( matFile, mat )

    return mat

# ***********************************************************************
# Cluster
# ***********************************************************************

X = []
for item in varNames:
    X.append( [ item ] )

model = AgglomerativeClustering( n_clusters = nClusters,
                                 linkage    = 'complete',
                                 affinity   = metric      )
model.fit( X )

# ***********************************************************************
# Echo results
# ***********************************************************************

labels = model.labels_
cltDf  = pd.DataFrame( { 'varName' : varNames, 'label' : labels } )

for label in np.unique( labels ):
    varNames = list( cltDf[ cltDf.label == label ].varName )
    print( 'Cluster', label, ':', varNames )

