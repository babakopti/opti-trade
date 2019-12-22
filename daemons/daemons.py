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

from daemonBase import Daemon, EmailTemplate

sys.path.append( os.path.abspath( '../' ) )

import utl.utils as utl

from mod.mfdMod import MfdMod
from prt.prt import MfdPrt

# ***********************************************************************
# Set some parameters 
# ***********************************************************************

indexes     = [ 'INDU', 'NDX', 'SPX', 'COMPQ', 'RUT',  'OEX',  
                'MID',  'SOX', 'RUI', 'RUA',   'TRAN', 'HGX',  
                'TYX',  'XAU'                      ] 
ETFs        = [ 'TQQQ', 'SPY', 'DDM', 'MVV', 'UWM', 'DIG', 'USD',
                'ERX',  'UYG', 'UPW', 'UGL', 'BIB', 'UST', 'UBT'  ]
futures     = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

DEV_LIST    = [ 'babak.emami@gmail.com' ]
USR_LIST    = [ 'babak.emami@gmail.com', 'farzin.shakib@gmail.com' ]

USR_EMAIL_TEMPLATE = 'templates/user_portfolio_email.txt'

DEBUG_MODE = True

if DEBUG_MODE:
    SCHED_FLAG = False
else:
    SCHED_FLAG = True

USE_OLD_DATA = False

NUM_DAYS_DATA = 2
NUM_DAYS_MOD  = 30
NUM_DAYS_PRT  = 730

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
                    nTrnDays    = 360,
                    nOosDays    = 3,
                    nPrdDays    = 1,
                    nMadDays    = 30,                    
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
                    schedTime   = '22:00',
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
        self.nMadDays    = nMadDays        
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

        if self.timeZone != 'America/New_York':
            self.logger.warning( 'Only America/New_York time zone is supported at this time!' )
            self.logger.warning( 'Switching to America/New_York time zone!' )
            self.timeZone = 'America/New_York'

        self.logger.info( 'Daemon is initialized ...' )            
            
    def setDfFile( self, snapDate ):

        self.logger.info( 'Getting data...' )
        
        nDays   = self.nTrnDays + self.nOosDays
        maxDate = pd.to_datetime( snapDate )
        minDate = maxDate - datetime.timedelta( days = nDays )
        
        if USE_OLD_DATA:
            try:
                oldDf = self.getOldDf()
            except Exception as e:
                oldDf = None
                self.logger.warning( e )            
                self.logger.warning( 'Could not read the old dfFile!' )
        else:
            oldDf = None
        
        updFlag = False
        if oldDf is not None:
            oldMinDate = list( oldDf.Date )[0]
            oldMaxDate = list( oldDf.Date )[-1]
            oldMinDate = pd.to_datetime( oldMinDate )
            oldMaxDate = pd.to_datetime( oldMaxDate )

            if minDate < oldMaxDate and \
               minDate.strftime( '%Y-%m-%d' ) > oldMinDate.strftime( '%Y-%m-%d' ) and \
               maxDate > oldMaxDate:
                self.logger.info( 'Updating the data file...' )
                updFlag = True
            else:
                self.logger.info( 'Something does not look right about the old data...' )
                self.logger.info( 'Not taking risk, will get all data from scratch...' )
                
        nDays = 0
        if updFlag:
            nDays = ( maxDate - oldMaxDate ).days
            
        if ( not updFlag ) or nDays <= 0:
            nDays = self.nTrnDays + self.nOosDays
        
        self.logger.info( 'Getting new data for last %d days...',
                          nDays )

        try:
            newDf = utl.getKibotData( etfs    = self.etfs,
                                      stocks  = self.stocks,
                                      futures = self.futures,
                                      indexes = self.indexes,
                                      nDays   = nDays       )
            self.logger.info( 'Done with getting new data!' )
        except Exception as e:
            self.logger.error( e )

        if updFlag:
            newDf = pd.concat( [ oldDf, newDf ] )
            
        fileName = 'dfFile_' + str( snapDate ) + '.pkl'
        filePath = os.path.join( self.datDir, fileName )

        try:
            newDf.to_pickle( filePath )
        except Exception as e:
            msgStr = e + '; Could not write the data file!'
            self.logger.error( msgStr )            

        self.dfFile = filePath

        try:
            self.checkDfSanity()
        except Exception as e:
            msgStr = e + '; Could not confirm the sanity of the dfFile!'
            self.logger.error( msgStr )
        
    def getOldDf( self ):

        tmpList = []
        tmpHash = {}
        pattern = 'dfFile_\d+-\d+-\d+ \d+:\d+:\d+.pkl'

        self.logger.info( 'Looking at the latest available data file...' )
        
        for fileName in os.listdir( self.datDir ):

            if not re.search( pattern, fileName ):
                continue

            baseName = os.path.splitext( fileName )[0]

            date = baseName.split( '_' )[1]

            date = pd.to_datetime( date )

            tmpList.append( fileName )
            
            tmpHash[ fileName ] = date

        if len( tmpList ) > 0:
            fileName = max( tmpList, key = lambda x : tmpHash[x] )
            filePath = os.path.join( self.datDir, fileName )
            
            self.logger.info( 'Found old data file %s...' % filePath )

            oldDf = pd.read_pickle( filePath )
        else:
            self.logger.info( 'No old data file found!' )
            self.logger.info( 'Will build a data file from scrach...' )
            return None

        oldDf[ 'Date' ] = oldDf[ 'Date' ].apply( pd.to_datetime )
        
        return oldDf
    
    def checkDfSanity( self ):

        self.logger.info( 'Checking sanity of the new data file...' )
        
        if not os.path.exists( self.dfFile ):
            msgStr =' Urgent: The file %s does not exist! Stopping the daemon...' %\
                self.dfFile
            self.logger.error( msgStr )
            self.stop()

        try:
            df = pd.read_pickle( self.dfFile )
        except Exception as e:
            msgStr = e + \
                '; Something is not right with the new data file %s!' %\
                self.dfFile
            self.logger.error( msgStr )

        items = [ 'Date' ] +\
            self.etfs +\
            self.stocks +\
            self.futures +\
            self.indexes

        for item in items:
            if not item in df.columns:
                msgStr =' Urgent: The file %s does not have a %s column! Stopping the daemon...' %\
                    ( self.dfFile, item )
                self.logger.error( msgStr )
                self.stop()

        self.logger.info( 'The new data file looks ok!' )
        
    def build( self ):

        os.environ[ 'TZ' ] = self.timeZone
        snapDate = datetime.datetime.now()
        snapDate = snapDate.strftime( '%Y-%m-%d %H:%M:%S' )
        snapDate = pd.to_datetime( snapDate )

        if not DEBUG_MODE:
            if snapDate.isoweekday() in [ 6 ]:
                return
        
        self.logger.info( 'Processing snapDate %s ...' % str( snapDate ) )

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
                           minTrnDate   = minTrnDt,
                           maxTrnDate   = maxTrnDt,
                           maxOosDate   = maxOosDt,
                           velNames     = self.velNames,
                           maxOptItrs   = self.maxOptItrs,
                           optGTol      = self.optTol,
                           optFTol      = self.optTol,
                           regCoef      = self.regCoef,
                           factor       = self.factor,
                           logFileName  = None,
                           verbose      = self.verbose      )

        try:
            sFlag = mfdMod.build()
        except Exception as e:
            msgStr = e + '; Model build was unsuccessful!'
            self.logger.error( msgStr )

        if sFlag:
            self.logger.info( 'Building model took %0.2f seconds!',
                              ( time.time() - t0 ) )
        else:
            self.logger.error( 'Model build was unsuccessful!' )
            self.logger.warning( 'Not building a portfolio based on this model!!' )
            return False

        mfdMod.save( modFile )

        if not os.path.exists( modFile ):
            self.logger.error( 'New model file is not written to disk!' )
            return False
            
        self.logger.info( 'Building portfolio for snapdate %s', str( snapDate ) )

        t0     = time.time()
        wtHash = {}
        curDt  = snapDate
        endDt  = snapDate + datetime.timedelta( days = self.nPrdDays )

        nPrdTimes = int( self.nPrdDays * 19 * 60 )

        mfdPrt = MfdPrt( modFile      = modFile,
                         assets       = self.assets,
                         nRetTimes    = self.nMadDays,
                         nPrdTimes    = nPrdTimes,
                         strategy     = 'mad',
                         minProbLong  = 0.5,
                         minProbShort = 0.5,
                         vType        = 'vel',
                         fallBack     = 'macd',
                         logFileName  = None,
                         verbose      = 1          )

        try:
            wtHash = mfdPrt.getPortfolio()
        except Exception as e:
            msgStr = e + '; Portfolio build was unsuccessful!'
            self.logger.error( msgStr )
            
        pickle.dump( wtHash, open( prtFile, 'wb' ) )    

        if not os.path.exists( modFile ):
            self.logger.error( 'New portfolio file is not written to disk!' )
            return False
        
        self.logger.info( 'Building portfolio took %0.2f seconds!',
                          ( time.time() - t0 ) )

        try:
            self.sendPrtAlert( wtHash )
        except Exception as e:
            msgStr = e + '; Portfolio alert was NOT sent!'
            self.logger.error( msgStr )

        try:
            self.clean( self, self.datDir, NUM_DAYS_DATA )
        except Exception as e:
            msgStr = e + '; Could not clean data directory!'
            self.logger.warning( msgStr )

        try:
            self.clean( self, self.modDir, NUM_DAYS_MOD )
        except Exception as e:
            msgStr = e + '; Could not clean model directory!'
            self.logger.warning( msgStr )

        try:
            self.clean( self, self.prtDir, NUM_DAYS_PRT )
        except Exception as e:
            msgStr = e + '; Could not clean portfolio directory!'
            self.logger.warning( msgStr )             
            
        return True

    def sendPrtAlert( self, wtHash ):

        assets = list( wtHash.keys() )
        percs  = []
        
        for asset in assets:
            percs.append( str( 100.0 * wtHash ) + '%' )

        df = pd.DataFrame( {  'Assets'     : assets,
                              'Percentage' : percs   } )
        
        pars[ 'Portfolio' ] = str( df )
            
        tempFile = open( USR_EMAIL_TEMPLATE, 'r' )
        tempStr  = tempFile.read()
        msgStr   = EmailTemplate( tempStr ).substitute( pars )

        tempFile.close()
        
        self.logger.critical( msgStr )

    def clean( self, fDir, nOldDays ):

        currTime = time.time()
        
        for fileName in os.listdir( fDir ):
            filePath = os.path.join( fDir, fileName )
            fileTime = os.path.getmtime( filePath )

            if fileTime < ( currTime - nOldDays * 86400 ):
                os.remove( filePath )
            
    def run( self ):

        os.environ[ 'TZ' ] = self.timeZone
        
        if not SCHED_FLAG:
            self.build()
        else:
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


