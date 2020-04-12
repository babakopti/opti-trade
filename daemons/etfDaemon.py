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
import json
import schedule
import re
import numpy as np
import pandas as pd
import pandas_market_calendars as pmc

from collections import defaultdict
from google.cloud import storage

from daemonBase import Daemon, EmailTemplate

sys.path.append( os.path.abspath( '../' ) )

import utl.utils as utl

from dat.assets import ETF_HASH, SUB_ETF_HASH, FUTURES
from mod.mfdMod import MfdMod
from prt.prt import MfdPrt

# ***********************************************************************
# Set some parameters 
# ***********************************************************************

INDEXES  = []
STOCKS   = []

ETFS     = list( ETF_HASH.keys() )

ASSETS   = list( SUB_ETF_HASH.keys() )

MAX_NUM_ASSETS = 5
NUM_ASSET_EVAL_DAYS = 60

NUM_TRN_DAYS  = 360
NUM_OOS_DAYS  = 3
NUM_PRD_DAYS  = 1
NUM_MAD_DAYS  = 30                    
MAX_OPT_ITRS  = 500
OPT_TOL       = 5.0e-2
REG_COEF      = 5.0e-3                    
FACTOR        = 4.0e-05
MOD_HEAD      = 'mfd_model_'
PRT_HEAD      = 'prt_weights_'                    
MOD_DIR       = '/var/mfd_models'
PRT_DIR       = '/var/prt_weights'
DAT_DIR       = '/var/mfd_data'
BASE_DAT_DIR  = '/var/data'
TIME_ZONE     = 'America/New_York'
SCHED_TIME    = '14:30'
LOG_FILE_NAME = '/var/log/mfd_prt_builder.log'
VERBOSE       = 1

PID_FILE      = '/var/run/mfd_prt_builder.pid'

USR_EMAIL_TEMPLATE = '/home/babak/opti-trade/daemons/templates/user_portfolio_email.txt'

DEV_LIST = [ 'babak.emami@gmail.com' ]
USR_LIST = [ 'babak.emami@gmail.com' ]

DEBUG_MODE = False

if DEBUG_MODE:
    SCHED_FLAG = False
else:
    SCHED_FLAG = True

NUM_DAYS_DATA_CLEAN = 2
NUM_DAYS_MOD_CLEAN  = 30
NUM_DAYS_PRT_CLEAN  = 730
NUM_DAYS_BUCKET_CLEAN = 5

GOOGLE_STORAGE_JSON = '/home/babak/opti-trade/daemons/keyfiles/google_storage.json'
GOOGLE_BUCKET = 'prt-storage'

# ***********************************************************************
# Class MfdPrtBuilder: Daemon to build portfolios using mfd, prt
# ***********************************************************************

class MfdPrtBuilder( Daemon ):

    def __init__(   self,
                    assets      = ASSETS,
                    etfs        = ETFS,
                    stocks      = STOCKS,
                    futures     = FUTURES,
                    indexes     = INDEXES,
                    maxAssets   = MAX_NUM_ASSETS,
                    nEvalDays   = NUM_ASSET_EVAL_DAYS,
                    nTrnDays    = NUM_TRN_DAYS,
                    nOosDays    = NUM_OOS_DAYS,
                    nPrdDays    = NUM_PRD_DAYS,
                    nMadDays    = NUM_MAD_DAYS,                    
                    maxOptItrs  = MAX_OPT_ITRS,
                    optTol      = OPT_TOL,
                    regCoef     = REG_COEF,                    
                    factor      = FACTOR,
                    modHead     = MOD_HEAD,
                    prtHead     = PRT_HEAD,
                    modDir      = MOD_DIR,
                    prtDir      = PRT_DIR,
                    datDir      = DAT_DIR,
                    baseDatDir  = BASE_DAT_DIR,
                    timeZone    = TIME_ZONE,
                    schedTime   = SCHED_TIME,
                    logFileName = LOG_FILE_NAME,
                    verbose     = VERBOSE         ):

        Daemon.__init__( self, PID_FILE )

        self.assets      = assets
        self.etfs        = etfs
        self.stocks      = stocks
        self.futures     = futures
        self.indexes     = indexes
        self.maxAssets   = maxAssets
        self.nEvalDays   = nEvalDays
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
        self.baseDatDir  = baseDatDir        
        self.timeZone    = timeZone
        self.schedTime   = schedTime
        self.logFileName = logFileName        
        self.verbose     = verbose
        self.velNames    = etfs + stocks + futures + indexes
        
        assert set( assets ).issubset( set( self.velNames ) ), \
            'Assets should be a subset of velNames!'

        assert self.maxAssets <= len( self.assets ), \
            'maxAssets should be <= number of assets in the pool!'

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
            
    def build( self ):

        os.environ[ 'TZ' ] = self.timeZone

        snapDate = datetime.datetime.now()
        snapDate = snapDate.strftime( '%Y-%m-%d %H:%M:%S' )
        snapDate = pd.to_datetime( snapDate )

        if not DEBUG_MODE:
            if snapDate.isoweekday() in [ 6, 7 ]:
                return

            holidays = pmc.get_calendar('NYSE').holidays().holidays
            
            if snapDate in holidays:
                self.logger.critical( 'Not running as today is a NYSE holiday!' )
                return            
        
        self.logger.info( 'Processing snapDate %s ...' % str( snapDate ) )

        try:
            self.setDfFile( snapDate )
        except Exception as e:
            self.logger.error( e )

        maxOosDt = snapDate
        maxTrnDt = maxOosDt - datetime.timedelta( days = self.nOosDays )
        minTrnDt = maxTrnDt - datetime.timedelta( days = self.nTrnDays )

        tmpStr   = snapDate.strftime( '%Y-%m-%d_%H:%M:%S' )
        modFile  = self.modHead + tmpStr + '.dill'
        prtFile  = self.prtHead + tmpStr + '.json'
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
            self.logger.error( 'The model did not converge!' )
            return False

        self.saveMod( mfdMod, modFile )

        if not os.path.exists( modFile ):
            self.logger.error( 'New model file is not written to disk!' )
            return False
            
        self.logger.info( 'Building portfolio for snapdate %s', str( snapDate ) )

        t0     = time.time()
        wtHash = {}

        nPrdTimes = int( self.nPrdDays * 19 * 60 )
        nRetTimes = int( self.nMadDays * 19 * 60 )

        quoteHash = self.getQuoteHash( mfdMod )
        
        mfdPrt = MfdPrt( modFile      = modFile,
                         quoteHash    = quoteHash,
                         nRetTimes    = nRetTimes,
                         nPrdTimes    = nPrdTimes,
                         strategy     = 'equal',
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

        self.savePrt( wtHash, prtFile )
            
        if not os.path.exists( prtFile ):
            self.logger.error( 'New portfolio file is not written to disk!' )
            return False
        
        self.logger.info( 'Building portfolio took %0.2f seconds!',
                          ( time.time() - t0 ) )

        try:
            self.sendPrtAlert( wtHash )
        except Exception as e:
            msgStr = e + '; Portfolio alert was NOT sent!'
            self.logger.error( msgStr )

        self.clean( self.datDir, NUM_DAYS_DATA_CLEAN )
        self.clean( self.modDir, NUM_DAYS_MOD_CLEAN  )
        self.clean( self.prtDir, NUM_DAYS_PRT_CLEAN  )

        self.cleanBucket( NUM_DAYS_BUCKET_CLEAN )
            
        return True

    def setDfFile( self, snapDate ):

        self.logger.info( 'Getting data...' )
        
        nDays   = self.nTrnDays + self.nOosDays
        maxDate = pd.to_datetime( snapDate )
        minDate = maxDate - datetime.timedelta( days = nDays )

        symbols = self.etfs + self.stocks + self.futures + self.indexes

        self.logger.info( 'Reading available data...' )
        
        oldDf = utl.mergeSymbols( symbols = symbols,
                                  datDir  = self.baseDatDir,
                                  fileExt = 'pkl',
                                  minDate = minDate,
                                  logger  = self.logger )

        self.logger.info( 'Getting new data...' )

        try:
            newDf = utl.getYahooData( etfs    = self.etfs,
                                      stocks  = self.stocks,
                                      futures = self.futures,
                                      indexes = self.indexes,
                                      nDays   = 5,
                                      logger  = self.logger  )
        
            self.logger.info( 'Done with getting new data!' )

            self.logger.info( 'Merging old and new data...' )
        
            newDf = newDf[ newDf.Date > oldDf.Date.max() ]
            newDf = pd.concat( [ oldDf, newDf ] )
        except Exception as e:
            msgStr = 'Could not get new data, old data max date is %s!' \
                     % str( oldDf.Date.max() )
            self.logger.error( msgStr )
            newDf = oldDf
            
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

    def saveMod( self, mfdMod, modFile ):

        try:
            mfdMod.save( modFile )
        except Exception as e:
            self.logger.error( e )

        if not os.path.exists( modFile ):
            self.logger.error( 'The model file was not generated!' )

        try:
            client   = storage.Client.from_service_account_json( GOOGLE_STORAGE_JSON )
            bucket   = client.get_bucket( GOOGLE_BUCKET )
            baseName = os.path.basename( modFile )
            blob     = bucket.blob( baseName )
            
            with open( modFile, 'rb' ) as fHd:
                blob.upload_from_file( fHd )

            self.logger.info( 'The model file %s was saved to bucket!',
                              baseName )

        except Exception as e:
            self.logger.error( e )

    def getQuoteHash( self, mfdMod ):

        ecoMfd    = mfdMod.ecoMfd
        snapDate  = ecoMfd.maxOosDate
        endDate   = snapDate
        begDate   = endDate - datetime.timedelta( days = self.nEvalDays )
        
        if self.maxAssets is None:
            assets = self.assets
        else:
            try:
                eDf = utl.sortAssets( symbols = self.assets,
                                      dfFile  = self.dfFile,
                                      begDate = begDate,
                                      endDate = endDate,
                                      logger  = self.logger     )
                assets = list( eDf.asset )[:self.maxAssets]
            except Exception as e:
                self.logger.error( e )

        quoteHash = {}
    
        for m in range( ecoMfd.nDims ):

            asset = ecoMfd.velNames[m]

            if asset not in assets:
                continue
        
            tmp       = ecoMfd.actOosSol[m][-1]
            slope     = ecoMfd.deNormHash[ asset ][0]
            intercept = ecoMfd.deNormHash[ asset ][1]
        
            quoteHash[ asset ] = slope * tmp + intercept

        return quoteHash
    
    def savePrt( self, wtHash, prtFile ):

        try:
            json.dump( wtHash, open( prtFile, 'w' ) )
        except Exception as e:
            self.logger.error( e )

        if not os.path.exists( prtFile ):
            self.logger.error( 'The portfolio file was not generated!' )

        try:
            client   = storage.Client.from_service_account_json( GOOGLE_STORAGE_JSON )
            bucket   = client.get_bucket( GOOGLE_BUCKET )
            baseName = os.path.basename( prtFile )
            blob     = bucket.blob( baseName )
            
            with open( prtFile, 'r' ) as fHd:
                tmpStr = str( json.load( fHd ) )
                blob.upload_from_string( tmpStr )

            self.logger.info( 'The portfolio file %s was saved to bucket!',
                              baseName )

        except Exception as e:
            self.logger.error( e )
            
    def sendPrtAlert( self, wtHash ):

        assets = list( wtHash.keys() )
        pars   = {}
        tmpStr = ''

        for asset in assets:
            perc    = 100.0 * wtHash[ asset ]
            tmpStr += '\n %10s: %0.2f %s\n' % ( asset, perc, '%' ) 

        pars[ 'Portfolio' ] = tmpStr

        tempFile = open( USR_EMAIL_TEMPLATE, 'r' )
        tempStr  = tempFile.read()
        msgStr   = EmailTemplate( tempStr ).substitute( pars )

        tempFile.close()

        self.logger.critical( msgStr )
        
        self.logger.info( 'Portfolio results sent to email lists!' )

    def clean( self, fDir, nOldDays ):

        self.logger.info( 'Cleaning up %s of files more than %d days old...',
                          fDir,
                          nOldDays )
        
        currTime = time.time()
        
        for fileName in os.listdir( fDir ):
            filePath = os.path.join( fDir, fileName )
            fileTime = os.path.getmtime( filePath )

            if fileTime < ( currTime - nOldDays * 86400 ):
                try:
                    os.remove( filePath )
                    self.logger.info( 'Deleted %s', filePath )
                except Exception as e:
                    self.logger.warning( e )

    def cleanBucket( self, nOldDays ):

        self.logger.info( 'Cleaning up cloud storage of files more than %d days old...',
                          nOldDays )
        
        tzObj    = pytz.timezone( self.timeZone )
        currTime = datetime.datetime.now( tzObj )
        client   = storage.Client.from_service_account_json( GOOGLE_STORAGE_JSON )
        bucket   = client.get_bucket( GOOGLE_BUCKET )        

        for blob in bucket.list_blobs():

            fileTime = blob.time_created

            if ( currTime - fileTime ).days > nOldDays:
                try:
                    blob.delete()
                    self.logger.info( 'Deleted %s from bucket %s',
                                      blob.name,
                                      bucket.name )
                except Exception as e:
                    self.logger.warning(e)

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


