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

numCores    = 2
modFlag     = True
dfFile      = 'data/dfFile_kibot.pkl'

nTrnDays    = 365
nOosDays    = 1
nPrdDays    = 3
bkBegDate   = pd.to_datetime( '2019-07-01 00:00:00' )
bkEndDate   = pd.to_datetime( '2019-12-19 23:59:00' )

indexes     = [ 'INDU', 'NDX', 'SPX', 'COMPQ', 'RUT',  'OEX',  
                'MID',  'SOX', 'RUI', 'RUA',   'TRAN', 'HGX',  
                'TYX',  'XAU'                      ] 

ETFs        = [ 'TQQQ', 'SPY', 'DDM', 'MVV', 'UWM', 'DIG', 'USD',
                'ERX',  'UYG', 'UPW', 'UGL', 'BIB', 'UST', 'UBT'  ]

invETFs     = [ 'SQQQ', 'SH',  'DXD', 'MZZ', 'TWM', 'DUG', 'SSG',
                'ERY',  'SKF', 'SDP', 'GLL', 'BIS', 'PST', 'TBT'  ]

futures     = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

velNames    = ETFs + futures + indexes

factor      = 4.0e-05
vType       = 'vel'
assets      = ETFs

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

    nPrdTimes   = int( nDays * 17 * 60 )

    mfdPrt = MfdPrt( modFile      = modFilePath,
                     assets       = assets,
                     nRetTimes    = 30,
                     nPrdTimes    = nPrdTimes,
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

    #os.remove( modFilePath )
    
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
    pool     = Pool( numCores )

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
    


