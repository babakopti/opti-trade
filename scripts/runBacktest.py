# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import datetime
import dill
import pickle
import numpy as np
import pandas as pd

from multiprocessing import Process, Pool

sys.path.append( os.path.abspath( '../' ) )

from mod.mfdMod import MfdMod
from prt.prt import MfdPrt 

# ***********************************************************************
# Set some parameters and read data
# ***********************************************************************

dfFilePath  = 'data/dfFile_2017plus.pkl'

nTrnDays    = 360
nPrdDays    = 7
bkBegDate   = pd.to_datetime( '2018-01-01 09:00:00' )
bkEndDate   = pd.to_datetime( '2018-12-31 17:00:00' )

indices     = [ 'INDU', 'NDX', 'SPX', 'COMPX', 'RUT',  'OEX',  
                'MID',  'SOX', 'RUI', 'RUA',   'TRAN', 'HGX',  
                'TYX',  'HUI', 'XAU'                       ] 

futures     = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

ETFs        = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH', 
                'SMH', 'XLE', 'XLF', 'XLU', 'EWJ'          ]

velNames    = indices + ETFs + futures

assets      = ETFs

totAssetVal = 1000000.0
tradeFee    = 6.95

# ***********************************************************************
# Utility functions
# ***********************************************************************

def buildModPrt( snapDate ):

    minTrnDt    = snapDate - datetime.timedelta( days = nTrnDays )
    maxTrnDt    = snapDate
    maxOosDt    = snapDate + datetime.timedelta( days = nPrdDays )
    modFilePath = 'models/model_' + str( snapDate ) + '.dill'
    wtFilePath  = 'models/weights_' + str( snapDate ) + '.pkl'

    print( 'Buiding model for snapdate', snapDate )

    t0     = time.time()

    mfdMod = MfdMod( dfFile       = dfFilePath,
                     minTrnDate   = minTrnDt,
                     maxTrnDate   = maxTrnDt,
                     maxOosDate   = maxOosDt,
                     velNames     = velNames,
                     maxOptItrs   = 500,
                     optGTol      = 3.0e-2,
                     optFTol      = 3.0e-2,
                     regCoef      = 1.0e-3,
                     minMerit     = 0.65,
                     maxBias      = 0.10,
                     varFiltFlag  = False,
                     validFlag    = False,
                     smoothCount  = None,
                     verbose      = 1          )

    sFlag = mfdMod.build()

    mfdMod.save( modFilePath )

    if sFlag:
        print( 'Buiding model took %d seconds!' % ( time.time() - t0 ) )
    else:
        print( 'Warning: Model did not converge!' )
        print( 'Warning: Not building a portfolio based on this model!!' )
        return False

    print( 'Buiding portfolio for snapdate', snapDate )

    t0        = time.time()
    ecoMfd    = mfdMod.ecoMfd
    quoteHash = {}
    wtHash    = {}

    for asset in assets:
        
        for m in range( ecoMfd.nDims ):
            if ecoMfd.velNames[m] == asset:
                break

        assert m < ecoMfd.nDims, \
            'Asset %s not found in the model!' % asset

        tmp       = ecoMfd.deNormHash[ asset ]
        slope     = tmp[0]
        intercept = tmp[1]
        price     = slope * ecoMfd.actSol[m][-1] + intercept
        
        quoteHash[ asset ] = price

    mfdPrt = MfdPrt( modFile      = modFilePath,
                     curDate      = maxTrnDt,
                     endDate      = maxOosDt, 
                     assets       = assets,
                     quoteHash    = quoteHash,
                     totAssetVal  = totAssetVal, 
                     tradeFee     = tradeFee,
                     strategy     = 'mad',
                     minProbLong  = 0.5,
                     minProbShort = 0.5,
                     verbose      = 1          )

    dateKey = snapDate.strftime( '%Y-%m-%d' )

    wtHash[ dateKey ] = mfdPrt.getPortfolio()

    pickle.dump( wtHash, open( wtFilePath, 'wb' ) )    
    
    print( 'Buiding portfolio took %d seconds!' % ( time.time() - t0 ) )

    return True

# ***********************************************************************
# Run the backtest
# ***********************************************************************
    
snapDate = bkBegDate
pool     = Pool()

while snapDate <= bkEndDate:

    while True:
        if snapDate.isoweekday() not in [ 6, 7 ]:
            break
        else:
            snapDate += datetime.timedelta( days = 1 )

    pool.apply_async( buildModPrt, args = ( snapDate, ) )

    snapDate = snapDate + datetime.timedelta( days = nPrdDays )

pool.close()
pool.join()
    


