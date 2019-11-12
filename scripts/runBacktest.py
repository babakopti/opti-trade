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

diffFlag    = False
modFlag     = True

if diffFlag:
    dfFile  = 'data/dfFile_2017plus_diff.pkl'
else:
    dfFile  = 'data/dfFile_2011_2014.pkl'

nTrnDays    = 360
nOosDays    = 3
nPrdDays    = 1
bkBegDate   = pd.to_datetime( '2014-01-01 00:00:00' )
bkEndDate   = pd.to_datetime( '2014-12-31 23:59:00' )

indices     = [ 'INDU', 'NDX', 'SPX', 'COMPX', 'RUT',  'OEX',  
                'MID',  'SOX', 'RUI', 'RUA',   'TRAN', 'HGX',  
                'TYX',  'HUI', 'XAU'                       ] 

futures     = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

ETFs        = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH', 
                'SMH', 'XLE', 'XLF', 'XLU', 'EWJ'          ]

allETFs     = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'GDX', 
                'OIH', 'RSX', 'SMH', 'XLE', 'XLF', 'XLV', 
                'XLU', 'FXI', 'TLT', 'EEM', 'EWJ', 'IYR', 
                'SDS', 'SLV', 'GLD', 'USO', 'UNG', 'TNA', 
                'TZA', 'FAS'                               ]

velNames    = indices + ETFs + futures

if diffFlag:
    nDims = len( velNames )
    for m in range( nDims ):
        velNames[m] = velNames[m] + '_Diff'

if diffFlag:
    factor = 1.0e-6
    vType  = 'var'
else:
    factor = 4.0e-05
    vType  = 'vel'

assets      = ETFs
totAssetVal = 1000000.0
tradeFee    = 6.95

# ***********************************************************************
# Utility functions
# ***********************************************************************

def buildModPrt( snapDate ):

    maxOosDt    = snapDate
    maxTrnDt    = maxOosDt - datetime.timedelta( days = nOosDays )
    minTrnDt    = maxTrnDt - datetime.timedelta( days = nTrnDays )
    modFilePath = 'models/model_' + str( snapDate ) + '.dill'
    wtFilePath  = 'models/weights_' + str( snapDate ) + '.pkl'

    if modFlag:
        print( 'Building model for snapdate', snapDate )

        t0     = time.time()

        mfdMod = MfdMod( dfFile       = dfFile,
                         minTrnDate   = minTrnDt,
                         maxTrnDate   = maxTrnDt,
                         maxOosDate   = maxOosDt,
                         velNames     = velNames,
                         maxOptItrs   = 50,
                         optGTol      = 5.0e-2,
                         optFTol      = 5.0e-2,
                         regCoef      = 1.0e-3,
                         factor       = factor,
                         verbose      = 1          )
        
        sFlag = mfdMod.build()

        if sFlag:
            print( 'Building model took %d seconds!' % ( time.time() - t0 ) )
        else:
            print( 'Warning: Model build was unsuccessful!' )
            print( 'Warning: Not building a portfolio based on this model!!' )
            return False

        mfdMod.save( modFilePath )
    else:
        mfdMod = dill.load( open( modFilePath, 'rb' ) )

    print( 'Building portfolio for snapdate', snapDate )

    t0     = time.time()
    wtHash = {}
    curDt  = snapDate
    endDt  = snapDate + datetime.timedelta( days = nPrdDays )
    nDays  = ( endDt - curDt ).days

    nPrdTimes   = int( nDays * 19 * 60 )

    mfdPrt = MfdPrt( modFile      = modFilePath,
                     assets       = assets,
                     nPrdTimes    = nPrdTimes,
                     totAssetVal  = totAssetVal, 
                     tradeFee     = tradeFee,
                     strategy     = 'mad',
                     minProbLong  = 0.5,
                     minProbShort = 0.5,
                     vType        = vType,
                     fallBack     = 'macd',
                     verbose      = 1          )

    dateKey = snapDate.strftime( '%Y-%m-%d' )

    wtHash[ dateKey ] = mfdPrt.getPortfolio()

    pickle.dump( wtHash, open( wtFilePath, 'wb' ) )    
    
    print( 'Building portfolio took %d seconds!' % ( time.time() - t0 ) )

    return True

# ***********************************************************************
# A worker function
# ***********************************************************************

def worker(snapDate):

    sFlag = False
    try:
        sFlag = buildModPrt( snapDate )
    except Exception as e:
        print( e )

    if not sFlag:
        print( 'ALERT: Processing of %s was unsuccessful!' % snapDate )

# ***********************************************************************
# Run the backtest
# ***********************************************************************

if __name__ ==  '__main__':
    snapDate = bkBegDate
    pool     = Pool()

    while snapDate <= bkEndDate:

        while True:
            if snapDate.isoweekday() not in [ 6, 7 ]:
                break
            else:
                snapDate += datetime.timedelta( days = 1 )

        pool.apply_async( worker, args = ( snapDate, ) )

        snapDate = snapDate + datetime.timedelta( days = nPrdDays )

    pool.close()
    pool.join()
    


