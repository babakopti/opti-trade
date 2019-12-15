# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import datetime
import pytz
import dill
import logging
import pickle
import schedule
import numpy as np
import pandas as pd

from multiprocessing import Process, Pool

from daemonBase import Daemon

sys.path.append( os.path.abspath( '../' ) )

from mod.mfdMod import MfdMod
from prt.prt import MfdPrt
from utl.utils import getLogger

# ***********************************************************************
# Set some parameters 
# ***********************************************************************

indices     = [ 'INDU', 'NDX', 'SPX', 'COMPX', 'RUT',  'OEX',  
                'MID',  'SOX', 'RUI', 'RUA',   'TRAN', 'HGX',  
                'TYX',  'HUI', 'XAU'                       ] 

futures     = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

ETFs        = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH', 
                'SMH', 'XLE', 'XLF', 'XLU', 'EWJ'          ]

velNames    = indices + ETFs + futures
assets      = ETFs

# ***********************************************************************
# Class MfdPrtBuilder: Daemon to build portfolios using mfd, prt
# ***********************************************************************

class MfdPrtBuilder( Daemon ):

    def __init__(   self,
                    assets      = assets,
                    velNames    = velNames,
                    nTrnDays    = 720,
                    nOosDays    = 3,
                    nPrdDays    = 1,
                    maxOptItrs  = 300,
                    optTol      = 5.0e-2,
                    regCoef     = 1.0e-3,                    
                    factor      = 4.0e-05,
                    modHead     = 'mfd_model_',
                    prtHead     = 'prt_weights_',                    
                    modDir      = '/var/mfd_models',
                    prtDir      = '/var/prt_weights',
                    datDir      = '/var/mfd_data',
                    timeZone    = 'America/New_York',
                    schedTime   = '10:00',
                    logFileName = '/var/log/mfd_prt_builder.log',
                    verbose     = 1         ):

        Daemon.__init__( self, '/var/run/mfd_prt_builder.pid' )
        
        assert set( assets ).issubset( set( velNames ) ), \
            'Assets should be a subset of velNames!'
        
        self.assets      = assets
        self.velNames    = velNames
        self.nTrnDays    = nTrnDays
        self.nOosDays    = nOosDays
        self.nPrdDays    = nPrdDays
        self.interval    = interval
        self.maxOptItrs  = maxOptItrs
        self.optTol      = optTol
        self.regCoef     = regCoef
        self.factor      = factor
        self.modHead     = modHead
        self.prtHead     = prtHead        
        self.modDir      = modDir
        self.prtDir      = prtDir
        self.datDir      = datDir
        self.timeZone    = timeZone
        self.schedTime   = schedTime
        self.logFileName = logFileName        
        self.verbose     = verbose

        if not os.path.exists( self.modDir ):
            os.makedirs( self.modDir )

        if not os.path.exists( self.prtDir ):
            os.makedirs( self.prtDir )

        if not os.path.exists( self.datDir ):
            os.makedirs( self.datDir )            
            
        self.logger = getLogger( logFileName, verbose )
        
        self.dfFile = None        

    def setDfFile( self, snapDate ):

        fileName = 'dfFile_' + str( snapDate )
        fileName = os.path.join( self.datDir, fileName )
        df = None
        df.to_pickle( fileName )
    
    def build( self ):

        os.environ[ 'TZ' ] = self.timeZone
        
        snapDate = datetime.datetime.now()
        
        if snapDate.isoweekday() in [ 6, 7 ]:
            return

        self.setDfFile( snapDate )
        
        maxOosDt = snapDate
        maxTrnDt = maxOosDt - datetime.timedelta( days = self.nOosDays )
        minTrnDt = maxTrnDt - datetime.timedelta( days = self.nTrnDays )

        tmpStr   = snapDate.strftime( '%Y-%m-%d_%H:%M:%S' )
        modFile  = self.modHead + tmpStr + '.dill'
        prtFile  = self.prtHead + tmpStr + '.pkl'
        modFile  = os.path.join( self.modDir, modFile )
        prtFile  = os.path.join( self.prtDir, prtFile )

        t0       = time.time()

        mfdMod   = MfdMod( dfFile       = self.dfFile,
                           minTrnDate   = self.minTrnDt,
                           maxTrnDate   = self.maxTrnDt,
                           maxOosDate   = self.maxOosDt,
                           velNames     = self.velNames,
                           maxOptItrs   = self.maxOptItrs,
                           optGTol      = self.optTol,
                           optFTol      = self.optTol,
                           regCoef      = self.regCoef,
                           factor       = self.factor,
                           verbose      = self.verbose      )
        
        sFlag = mfdMod.build()

        if sFlag:
            self.logger.info( 'Building model took %0.2f seconds!',
                              ( time.time() - t0 ) )
        else:
            self.logger.critical( 'Model build was unsuccessful!' )
            self.logger.warning( 'Not building a portfolio based on this model!!' )
            return False

        mfdMod.save( modFile )

        self.logger.info( 'Building portfolio for snapdate %s', str( snapDate ) )

        t0     = time.time()
        wtHash = {}
        curDt  = snapDate
        endDt  = snapDate + datetime.timedelta( days = self.nPrdDays )

        nPrdTimes = int( self.nPrdDays * 19 * 60 )

        mfdPrt = MfdPrt( modFile      = modFile,
                         assets       = assets,
                         nRetTimes    = 30,
                         nPrdTimes    = nPrdTimes,
                         strategy     = 'mad',
                         minProbLong  = 0.5,
                         minProbShort = 0.5,
                         vType        = 'vel',
                         fallBack     = 'macd',
                         verbose      = 1          )

        wtHash  = mfdPrt.getPortfolio()

        pickle.dump( wtHash, open( prtFile, 'wb' ) )    
    
        self.logger.info( 'Building portfolio took %0.2f seconds!',
                          ( time.time() - t0 ) )

        return True

    def run( self ):

        os.environ[ 'TZ' ] = self.timeZone

        schedule.every().day.at( self.schedTime ).do( self.build )

        while True: 
              schedule.run_pending() 
              time.sleep( 60 )
              
# ***********************************************************************
# Run daemon
# ***********************************************************************

if __name__ ==  '__main__':

    daemon = MfdPrtBuilder()
    
    if len(sys.argv) == 2:
        if 'start' == sys.argv[1]:
            daemon.start()
        elif 'stop' == sys.argv[1]:
            daemon.stop()
        elif 'restart' == sys.argv[1]:
            daemon.restart()
        else:
            print( 'Unknown command' )
            sys.exit(2)
        sys.exit(0)
    else:
        print( 'usage: %s start|stop|restart' % sys.argv[0] )
        sys.exit(2)


