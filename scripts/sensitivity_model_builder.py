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

snapDates   = [ '2018-03-10', '2019-05-06', '2017-07-23' ]

# snapDates   = [ '2018-01-15', '2018-02-15', '2019-04-18',
#                 '2018-05-15', '2018-06-15', '2019-07-18',
#                 '2018-08-15', '2018-09-18', '2019-10-15',
#                 '2018-12-18' ]

# trnDaysList = [ 360 ]
# tolList     = [ 0.1, 0.05, 0.03, 0.01, 0.005, 0.001 ]
# regCoefList = [ 1.0e-2 ]
# atnFctList  = [ 1.0 ]

# trnDaysList = [ 360 ]
# tolList     = [ 0.05 ]
# regCoefList = [ 1.0e-5, 1.0e-4, 1.0e-1 ]
# atnFctList  = [ 1.0 ]

# trnDaysList = [ 1080, 1170, 1260, 1350, 1440 ]
# tolList     = [ 0.05 ]
# regCoefList = [ 1.0e-3 ]
# atnFctList  = [ 1.0 ]

trnDaysList = [ 365 * 5, 365 * 10, 365 * 15 ]
tolList     = [ 0.01 ]
regCoefList = [ 1.0e-3 ]
atnFctList  = [ 1.0 ]

modDir      = 'models_sensitivity_long_term'
dfFile      = 'data/dfFile_long_term_all.pkl'
nOosDays    = 6 * 30
maxOptItrs  = 500

indexes = [ 'INDU', 'NDX', 'SPX', 'RUT', 'OEX',
            'MID',  'SOX', 'RUI', 'RUA', 'TRAN',
            'HGX',  'TYX', 'XAU' ]

futures = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

ETFs    = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH',
            'SMH', 'XLE', 'XLF', 'XLU', 'EWJ' ]

velNames    = indexes + ETFs + futures

factor      = 1.0e-05
numCores    = 1

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
    
    pool = Pool( numCores )

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
                        
    build()

                    
