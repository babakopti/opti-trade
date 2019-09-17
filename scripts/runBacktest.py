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

sys.path.append( os.path.abspath( '../' ) )

from mod.mfdMod import MfdMod
from prt.prt import MfdPrt 

# ***********************************************************************
# Set some parameters and read data
# ***********************************************************************

dfFilePath  = 'data/dfFile.pkl'

nTrnDays    = 21
nPrdDays    = 7
bkBegDate   = pd.to_datetime( '2019-06-01 09:00:00' )
bkEndDate   = pd.to_datetime( '2019-07-31 17:00:00' )

indices     = [ 'INDU', 'NDX', 'SPX', 'COMPX', 'RUT',  'OEX',  
                'MID',  'SOX', 'RUI', 'RUA',   'TRAN', 'HGX',  
                'TYX',  'HUI', 'XAU'                       ] 

futures     = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

ETFs        = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH', 
                'SMH', 'XLE', 'XLF', 'XLU', 'EWJ'          ]

velNames    = indices + ETFs 

assets      = ETFs

totAssetVal = 1000000.0
tradeFee    = 6.95
wtFilePath  = 'weights.pkl'

# ***********************************************************************
# Run the backtest
# ***********************************************************************
    
snapDate      = bkBegDate
wtHash        = {}

while snapDate <= bkEndDate:

    while True:
        if snapDate.isoweekday() not in [ 6, 7 ]:
            break
        else:
            snapDate += datetime.timedelta( days = 1 )

    print( 'Buiding model for snapdate', snapDate )

    t0           = time.time()
    _minTrnDt    = snapDate - datetime.timedelta( days = nTrnDays )
    _maxTrnDt    = snapDate
    _maxOosDt    = snapDate + datetime.timedelta( days = nPrdDays )
    _modFilePath = 'models/model_' + str( snapDate ) + '.dill'

    mfdMod = MfdMod( dfFile       = dfFilePath,
                     minTrnDate   = _minTrnDt,
                     maxTrnDate   = _maxTrnDt,
                     maxOosDate   = _maxOosDt,
                     velNames     = velNames,
                     maxOptItrs   = 200,
                     optGTol      = 1.0e-2,
                     optFTol      = 1.0e-2,
                     regCoef      = 1.0e-5,
                     minMerit     = 0.65,
                     maxBias      = 0.10,
                     varFiltFlag  = False,
                     validFlag    = False,
                     smoothCount  = None,
                     verbose      = 1          )

    validFlag = mfdMod.build()

    mfdMod.save( _modFilePath )

    if validFlag:
        print( 'Buiding model took %d seconds!' % ( time.time() - t0 ) )
    else:
        print( 'Warning: Model did not converge!' )

    print( 'Buiding portfolio for snapdate', snapDate )

    t0         = time.time()
    ecoMfd     = mfdMod.ecoMfd
    _quoteHash = {}

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
        
        _quoteHash[ asset ] = price

    mfdPrt = MfdPrt( modFile      = _modFilePath,
                     curDate      = _maxTrnDt,
                     endDate      = _maxOosDt, 
                     assets       = assets,
                     quoteHash    = _quoteHash,
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

    snapDate = _maxOosDt


