# ***********************************************************************
# Import libraries
# ***********************************************************************

import os
import sys
import dill
import time
import datetime
import random
import numpy as np
import pandas as pd

from multiprocessing import Process, Pool

sys.path.append( os.path.abspath( '../' ) )

from mod.mfdMod import MfdMod

# ***********************************************************************
# Set some parameters
# ***********************************************************************

snapDates   = [ '2018-03-10', '2018-11-05', '2019-05-10' ]

# trnDaysList = [ 360 ]
# tolList     = [ 0.1, 0.05, 0.03, 0.01, 0.005, 0.001 ]
# regCoefList = [ 1.0e-2 ]
# atnFctList  = [ 1.0 ]

# trnDaysList = [ 360 ]
# tolList     = [ 0.05 ]
# regCoefList = [ 1.0e-5, 1.0e-4, 1.0e-1 ]
# atnFctList  = [ 1.0 ]

# trnDaysList = [ 60, 30 ]
# tolList     = [ 0.05 ]
# regCoefList = [ 1.0e-3 ]
# atnFctList  = [ 1.0 ]

trnDaysList = [ 360 ]
tolList     = [ 0.05 ]
regCoefList = [ 1.0e-3 ]
atnFctList  = [ 1.0, 0.99, 0.95, 0.9, 0.8 ]

modFlag     = True
modDir      = 'models_sensitivity'
dfFile      = 'data/dfFile_2017plus.pkl'
nOosDays    = 3
maxOptItrs  = 300

indices     = [ 'INDU', 'NDX', 'SPX', 'COMPX', 'RUT',  'OEX',  
                'MID',  'SOX', 'RUI', 'RUA',   'TRAN', 'HGX',  
                'TYX',  'HUI', 'XAU'                       ] 
futures     = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]
ETFs        = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH', 
                'SMH', 'XLE', 'XLF', 'XLU', 'EWJ'          ]
velNames    = indices + ETFs + futures

factor      = 4.0e-05

# ***********************************************************************
# Some utility functions
# ***********************************************************************

def getFilePath( snapDate, nTrnDays, tol, regCoef, atnFct ):

    modFileName = 'model_' +\
        str( snapDate ) +\
        '_nTrnDays_' + str( nTrnDays ) +\
        '_tol_' + str( tol ) +\
        '_regCoef_' + str( regCoef ) +\
        '_atnFct_' + str( atnFct ) +\
        '.dill'
    
    modFilePath = os.path.join( modDir, modFileName )

    return modFilePath

def buildMod( snapDate, nTrnDays, tol, regCoef, atnFct ):
    
    maxOosDt = pd.to_datetime( snapDate )
    maxTrnDt = maxOosDt - datetime.timedelta( days = nOosDays )
    minTrnDt = maxTrnDt - datetime.timedelta( days = nTrnDays )

    modFilePath = getFilePath( snapDate,
                               nTrnDays,
                               tol,
                               regCoef,
                               atnFct   )

    print( 'Building a model for snapdate', snapDate )

    t0     = time.time()

    mfdMod = MfdMod( dfFile       = dfFile,
                     minTrnDate   = minTrnDt,
                     maxTrnDate   = maxTrnDt,
                     maxOosDate   = maxOosDt,
                     velNames     = velNames,
                     maxOptItrs   = maxOptItrs,
                     optGTol      = tol,
                     optFTol      = tol,
                     regCoef      = regCoef,
                     factor       = factor,
                     atnFct       = atnFct,
                     verbose      = 1          )

    sFlag = mfdMod.build()

    if sFlag:
        print( 'Building model took %d seconds!' % ( time.time() - t0 ) )
        mfdMod.save( modFilePath )
    else:
        print( 'Warning: Model build was unsuccessful!' )

    return

def build():
    
    pool = Pool()

    for snapDate in snapDates:
        for nTrnDays in trnDaysList:
            for tol in tolList:
                for regCoef in regCoefList:
                    for atnFct in atnFctList:
                        
                        argTuple = ( snapDate,
                                     nTrnDays,
                                     tol,
                                     regCoef,
                                     atnFct  )

                        pool.apply_async( buildMod, args = argTuple )

    pool.close()
    pool.join()

# ***********************************************************************
# Loop through parameters, build models, and generate plots
# ***********************************************************************

if __name__ ==  '__main__':
                        
    if modFlag:
        build()

                    
