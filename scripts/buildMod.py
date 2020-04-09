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

dfFile      = 'data/dfFile_2016plus.pkl'

minTrnDate  = pd.to_datetime( '2017-02-01 09:00:00' )
maxTrnDate  = pd.to_datetime( '2018-01-31 09:00:00' )
maxOosDate  = pd.to_datetime( '2018-02-10 23:59:00' )

indices     = [ 'INDU', 'NDX', 'SPX', 'RUT', 'OEX',  
                'MID',  'SOX', 'RUI', 'RUA', 'TRAN',
                'HGX',  'TYX', 'XAU'               ] 

futures     = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

ETFs        = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH', 
                'SMH', 'XLE', 'XLF', 'XLU', 'EWJ'          ]

velNames    = ETFs + indices + futures

pType       = 'vel'

modFileName = 'models/model.dill'

# ***********************************************************************
# Build model
# ***********************************************************************

mfdMod = MfdMod(    dfFile       = dfFile,
                    minTrnDate   = minTrnDate,
                    maxTrnDate   = maxTrnDate,
                    maxOosDate   = maxOosDate,
                    velNames     = velNames,
                    optType      = 'SLSQP',
                    maxOptItrs   = 100,
                    optGTol      = 1.0e-2,
                    optFTol      = 1.0e-2,
                    factor       = 4.0e-5,
                    regCoef      = 1.0e-3,
                    diagFlag     = True,
                    elastFlag    = False,
                    elastCoef    = 0.05,
                    smoothCount  = None,
                    logFileName  = None,
                    verbose      = 1          )

validFlag = mfdMod.build()

print( 'Success :', validFlag )

mfdMod.save( modFileName )
#mfdMod.ecoMfd.pltResults( rType = 'trn', pType = pType )
#mfdMod.ecoMfd.pltResults( rType = 'oos', pType = pType )

gammaId = 0
nDims = mfdMod.ecoMfd.nDims

for r in range( nDims ):
    for p in range( nDims ):
        for q in range( p, nDims ):

            if r != p and r != q and p != q:
                continue
                    
            if r == p and r == q:
                print(mfdMod.ecoMfd.GammaVec[gammaId])

            gammaId += 1
