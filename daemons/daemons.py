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
import re
import numpy as np
import pandas as pd

from multiprocessing import Process, Pool

from daemonBase import Daemon

sys.path.append( os.path.abspath( '../' ) )

import utl.utils as utl

from mod.mfdMod import MfdMod
from prt.prt import MfdPrt

# ***********************************************************************
# Set some parameters 
# ***********************************************************************

indexes     = [ 'INDU', 'NDX', 'SPX', 'COMPX', 'RUT',  'OEX',  
                'MID',  'SOX', 'RUI', 'RUA',   'TRAN', 'HGX',  
                'TYX',  'HUI', 'XAU'                       ] 

futures     = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

ETFs        = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH', 
                'SMH', 'XLE', 'XLF', 'XLU', 'EWJ'          ]

DEV_LIST    = [ 'babak.emami@gmail.com' ]
USR_LIST    = [ 'babak.emami@gmail.com' ]

# ***********************************************************************
# Class MfdPrtBuilder: Daemon to build portfolios using mfd, prt
# ***********************************************************************

class MfdPrtBuilder( Daemon ):

    def __init__(   self,
                    assets      = ETFs,
                    etfs        = ETFs,
                    stocks      = [],
                    futures     = futures,
                    indexes     = indexes,
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

        self.assets      = assets
        self.etfs        = etfs
        self.stocks      = stocks
        self.futures     = futures
        self.indexes     = indexes
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
        self.velNames    = etfs + stocks + futures + indexes
        
        assert set( assets ).issubset( set( self.velNames ) ), \
            'Assets should be a subset of velNames!'

        if not os.path.exists( self.modDir ):
            os.makedirs( self.modDir )

        if not os.path.exists( self.prtDir ):
            os.makedirs( self.prtDir )

        if not os.path.exists( self.datDir ):
            os.makedirs( self.datDir )            
            
        self.logger = utl.getLogger( logFileName, verbose )

        devAlertHd = utl.getAlertHandler( alertLevel = logging.ERROR,
                                          subject    = 'A message for Opti-Trade developers!',
                                          mailList   = DEV_LIST )
        
        usrAlertHd = utl.getAlertHandler( alertLevel = logging.CRITICAL,
                                          subject    = 'A message for Opti-Trade users!',
                                          mailList   = USR_LIST )

        self.logger.addHandler( devAlertHd )
        self.logger.addHandler( usrAlertHd )
        
        self.dfFile = None        

    def setDfFile( self, snapDate ):

        nDays   = self.nTrnDays + self.nOosDays
        maxDate = pd.to_datetime( snapDate )
        minDate = maxdate - datetime.timedelta( days = nDays )

        try:
            oldDf = self.getOldDf()
        except Exception as e:
            oldDf = None
            self.logger.warning( e )            
            self.logger.warning( 'Could not read the old dfFile!' )

        updFlag = False
        
        if oldDf is not None:
            
            oldMinDate = min( oldDf.Date )
            oldMaxDate = max( oldDf.Date )
            
            if minDate < oldMaxDate and \
               minDate > oldMinDate and \
               maxDate > oldMaxDate:
                updFlag = True
                
        if updFlag:
            nDays = ( maxDate - oldMaxDate ).days
        else:
            nDays = self.nTrnDays + self.nOosDays

        newDf = utl.getKibotData( etfs    = self.etfs,
                                  stocks  = self.stocks,
                                  futures = self.futures,
                                  indexes = self.indexes,
                                  nDays   = nDays       )

        if updFlag:
            newDf = pd.concat( [ oldDf, newDf ] )
            
        fileName = 'dfFile_' + str( snapDate )
        filePath = os.path.join( self.datDir, fileName )

        try:
            newDf.to_pickle( filePath )
        except Exception as e:
            msgStr = e + '; Could not write the data file!'
            self.logger.error( msgStr )            

        self.dfFile = filePath

        try:
            df = read_pickle( self.dfFile )
        except Exception as e:
            msgStr = e + \
                '; Something is not right with the new data file %s!' %\
                self.dfFile
            self.logger.error( msgStr )        

    def getOldDf( self ):

        tmpList = []
        tmpHash = {}
        pattern = 'dfFile_\d+-\d+-\d+ \d+:\d+:\d+.pkl'
        
        for fileName in os.listdir( self.datDir ):

            if not re.search( pattern, item ):
                continue

            baseName = os.path.splitext( item )
            date = baseName.split( '_' )[1]
            date = pd.to_datetime( date )

            tmpList.append( fileName )
            
            tmpHash[ fileName ] = date

        if len( tmpList ) > 0:
            fileName = max( tmpList, key = lambda x : tmpHash[x] )
            filePath = os.path.join( self.datDir, fileName )
            oldDf    = df.read_pickle( filePath )
        else:
            return None

        return oldDf
    
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


