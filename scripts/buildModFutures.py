# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import datetime
import numpy as np
import pandas as pd

from utils import getDf

sys.path.append( os.path.abspath( '../' ) )

from mod.mfdMod import MfdMod

# ***********************************************************************
# Set some parameters and read data
# ***********************************************************************

dfFile  = 'data/dfFile_futures.pkl'

minTrnDate  = pd.to_datetime( '2016-01-01 09:00:00' )
maxTrnDate  = pd.to_datetime( '2019-06-30 23:59:00' )
maxOosDate  = pd.to_datetime( '2019-12-25 23:59:00' )

indexes = [ 'INDU', 'NDX', 'SPX', 'COMPQ', 'RUT',  'OEX',  
            'MID',  'SOX', 'RUI', 'RUA',   'TRAN', 'HGX',  
            'TYX'                      ] 

# fuDf = pd.read_csv( 'data/Futures_kibot.txt', delimiter = '\t' )

# fuDf[ 'Continuous' ] = fuDf.Description.apply( lambda x : 'CONTINUOUS' in x )

# fuDf = fuDf[ fuDf.Continuous == True ]

# fuDf = fuDf[ [ 'Base', 'StartDate', 'Description' ] ]

# fuDf[ 'StartDate' ] = fuDf[ 'StartDate' ].apply( pd.to_datetime )

# fuDf.reset_index( drop = True, inplace = True )

# fuDf[ fuDf.StartDate <= pd.to_datetime( '2010-01-01' ) ].shape

# futures = list( set( fuDf.Base ) - set( [ 'RTY', 'TN', 'BTC', 'SIR', 'SIL'  ] ) )

futures     = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM', 'CL', 'NG',
                'GC', 'SI', 'TY', 'FV', 'TU', 'C', 'HG', 'S', 'W', 'RB',
                'BO', 'O' ]

velNames    = futures + indexes

pType = 'vel'

modFileName = 'models/model_futures.dill'

factor = 1.0e-5
    
# ***********************************************************************
# Build model
# ***********************************************************************

mfdMod = MfdMod(    dfFile       = dfFile,
                    minTrnDate   = minTrnDate,
                    maxTrnDate   = maxTrnDate,
                    maxOosDate   = maxOosDate,
                    velNames     = velNames,
                    maxOptItrs   = 300,
                    optGTol      = 5.0e-2,
                    optFTol      = 5.0e-2,
                    factor       = factor,
                    regCoef      = 1.0e-3,
                    selParams    = None,
                    atnFct       = 1.0,
                    verbose      = 1          )

validFlag = mfdMod.build()

print( 'Success :', validFlag )

mfdMod.save( modFileName )

#mfdMod.ecoMfd.pltResults( rType = 'trn', pType = pType )
#mfdMod.ecoMfd.pltResults( rType = 'oos', pType = pType )

