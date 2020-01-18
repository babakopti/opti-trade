# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import datetime
import dill
import pickle
import logging
import numpy as np
import pandas as pd

from multiprocessing import Process, Pool

sys.path.append( os.path.abspath( '../' ) )

import utl.utils as utl

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
    dfFile  = 'data/dfFile_kibot_2016plus.pkl'

nTrnDays    = 360
nOosDays    = 3
nPrdDays    = 1
bkBegDate   = pd.to_datetime( '2018-10-26 09:00:00' )
bkEndDate   = pd.to_datetime( '2018-12-31 09:00:00' )

# indices     = [ 'INDU', 'NDX', 'SPX', 'COMPX', 'RUT',  'OEX',  
#                 'MID',  'SOX', 'RUI', 'RUA',   'TRAN', 'HGX',  
#                 'TYX',  'HUI', 'XAU'                       ] 
# indices     = [ 'INDU', 'NDX', 'SPX', 'COMPQ', 'RUT',  'OEX',  
#                 'MID',  'SOX', 'RUI', 'RUA',   'TRAN', 'HGX',  
#                 'TYX',  'XAU'                      ]

futures     = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

# ETFs        = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH', 
#                 'SMH', 'XLE', 'XLF', 'XLU', 'EWJ'          ]
#ETFs        = [ 'TQQQ', 'SPY', 'DDM', 'MVV', 'UWM', 'DIG', 'USD',
#                'ERX',  'UYG', 'UPW', 'UGL', 'BIB', 'UST', 'UBT'  ]
# allETFs     = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'GDX', 
#                 'OIH', 'RSX', 'SMH', 'XLE', 'XLF', 'XLV', 
#                 'XLU', 'FXI', 'TLT', 'EEM', 'EWJ', 'IYR', 
#                 'SDS', 'SLV', 'GLD', 'USO', 'UNG', 'TNA', 
#                 'TZA', 'FAS'                               ]

allETFs     = [ 'TQQQ', 'SPY', 'DDM', 'MVV', 'UWM',  'SAA',
                'UYM',  'UGE', 'UCC', 'FINU', 'RXL', 'UXI',
                'URE',  'ROM', 'UJB', 'AGQ',  'DIG', 'USD',
                'ERX',  'UYG', 'UCO', 'BOIL', 'UPW', 'UGL',
                'BIB', 'UST', 'UBT'  ]

velNames    = allETFs + futures

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

#assets      = ETFs

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

        logging.shutdown()
        
        mfdMod = MfdMod( dfFile       = dfFile,
                         minTrnDate   = minTrnDt,
                         maxTrnDate   = maxTrnDt,
                         maxOosDate   = maxOosDt,
                         velNames     = velNames,
                         maxOptItrs   = 500,
                         optGTol      = 5.0e-2,
                         optFTol      = 5.0e-2,
                         regCoef      = 5.0e-3,
                         factor       = factor,
                         logFileName  = None,
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

    nPrdTimes = int( nDays * 19 * 60 )
    nRetTimes = int( 30 * 17 * 60 )  

    assets = []

    eDf = utl.sortAssets( allETFs, 90 )
    assets = list( eDf.asset )[:5]
    
    # perfs  = mfdMod.ecoMfd.getOosTrendPerfs( 'vel' )
    # for i in range( mfdMod.ecoMfd.nDims ):
    #     if mfdMod.ecoMfd.velNames[i] in allETFs and perfs[i]:
    #         assets.append( mfdMod.ecoMfd.velNames[i] )
    
    mfdPrt = MfdPrt( modFile      = modFilePath,
                     assets       = assets,
                     nRetTimes    = nRetTimes,
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
    


