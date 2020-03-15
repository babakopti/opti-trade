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

from collections import defaultdict

from google.cloud import storage

from daemonBase import Daemon, EmailTemplate

sys.path.append( os.path.abspath( '../' ) )

import utl.utils as utl

from dat.assets import ETF_HASH, SUB_ETF_HASH, FUTURES
from mod.mfdMod import MfdMod
from prt.prt import MfdOptionsPrt
from brk.tdam import Tdam

# ***********************************************************************
# Set some parameters 
# ***********************************************************************

INDEXES  = []
STOCKS   = []

ETFS     = list( ETF_HASH.keys() )

ASSETS   = list( SUB_ETF_HASH.keys() )

NUM_TRN_YEARS = 10
MAX_OPT_ITRS  = 500
OPT_TOL       = 1.0e-2
REG_COEF      = 1.0e-3                    
FACTOR        = 1.0e-5

MAX_OPTION_MONTHS  = 6
MAX_RATIO_CONTRACT = 0.5
MAX_RATIO_ASSET    = 0.5
MAX_EXPOSURE_RATIO = 0.5
MIN_PROBABILITY    = 0.7
OPTION_TRADE_FEE   = 0.5

MOD_HEAD      = 'option_model_'
PRT_HEAD      = 'option_prt_'                    
MOD_DIR       = '/var/option_models'
PRT_DIR       = '/var/option_prt'
DAT_DIR       = '/var/option_data'
BASE_DAT_DIR  = '/var/data'
TIME_ZONE     = 'America/New_York'
SCHED_TIME    = '04:00'
LOG_FILE_NAME = '/var/log/option_prt_builder.log'
VERBOSE       = 1

PID_FILE      = '/var/run/option_prt_builder.pid'

USR_EMAIL_TEMPLATE = '/home/babak/opti-trade/daemons/templates/user_portfolio_email_option.txt'

DEV_LIST = [ 'babak.emami@gmail.com' ]
USR_LIST = [ 'babak.emami@gmail.com' ]

TOKEN_FILE = '../brk/tokens/refresh_token.txt'

with open( TOKEN_FILE, 'r' ) as fHd:
    REFRESH_TOKEN = fHd.read()[:-1]

DEBUG_MODE = False

if DEBUG_MODE:
    SCHED_FLAG = False
else:
    SCHED_FLAG = True

NUM_DAYS_DATA_CLEAN = 2
NUM_DAYS_MOD_CLEAN  = 30
NUM_DAYS_PRT_CLEAN  = 730
NUM_DAYS_BUCKET_CLEAN = 5

# ***********************************************************************
# Class OptionPrtBuilder: Daemon to build options portfolios
# ***********************************************************************

class OptionPrtBuilder( Daemon ):

    def __init__(   self,
                    assets      = ASSETS,
                    etfs        = ETFS,
                    stocks      = STOCKS,
                    futures     = FUTURES,
                    indexes     = INDEXES,
                    nTrnYears   = NUM_TRN_YEARS,
                    maxOptItrs  = MAX_OPT_ITRS,
                    optTol      = OPT_TOL,
                    regCoef     = REG_COEF,                    
                    factor      = FACTOR,
                    maxMonths   = MAX_OPTION_MONTHS,
                    maxRatioC   = MAX_RATIO_CONTRACT,
                    maxRatioA   = MAX_RATIO_ASSET,
                    maxExpRatio = MAX_EXPOSURE_RATIO,
                    minProb     = MIN_PROBABILITY,
                    tradeFee    = OPTION_TRADE_FEE,
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
        self.nTrnYears   = nTrnYears
        self.maxOptItrs  = maxOptItrs
        self.optTol      = optTol
        self.regCoef     = regCoef
        self.factor      = factor
        self.maxMonths   = maxMonths
        self.maxRatioC   = maxRatioC
        self.maxRatioA   = maxRatioA
        self.maxExpRatio = MAX_EXPOSURE_RATIO
        self.minProb     = minProb
        self.tradeFee    = tradeFee
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

        if not os.path.exists( self.modDir ):
            os.makedirs( self.modDir )

        if not os.path.exists( self.prtDir ):
            os.makedirs( self.prtDir )

        if not os.path.exists( self.datDir ):
            os.makedirs( self.datDir )            
            
        self.logger = utl.getLogger( logFileName, verbose )

        devAlertHd = utl.getAlertHandler( alertLevel = logging.ERROR,
                                          subject    = 'A message about options selection!',
                                          mailList   = DEV_LIST )
        
        usrAlertHd = utl.getAlertHandler( alertLevel = logging.CRITICAL,
                                          subject    = 'A message about options selection!',
                                          mailList   = USR_LIST )

        self.logger.addHandler( devAlertHd )
        self.logger.addHandler( usrAlertHd )
        
        self.dfFile = None        

        if self.timeZone != 'America/New_York':
            self.logger.warning( 'Only America/New_York time zone is supported at this time!' )
            self.logger.warning( 'Switching to America/New_York time zone!' )
            self.timeZone = 'America/New_York'

        self.logger.info( 'Daemon is initialized ...' )            
            
    def process( self ):

        os.environ[ 'TZ' ] = self.timeZone

        snapDate = datetime.datetime.now()
        snapDate = snapDate.strftime( '%Y-%m-%d %H:%M:%S' )
        snapDate = pd.to_datetime( snapDate )

        if not DEBUG_MODE:
            if snapDate.isoweekday() in [ 6, 7 ]:
                return
        
        self.logger.info( 'Processing snapDate %s ...' % str( snapDate ) )

        try:
            self.setDfFile( snapDate )
        except Exception as e:
            self.logger.error( e )

        try:
            self.buildMod( snapDate )
        except Exception as e:            
            msgStr = e + '; Model build was unsuccessful!'
            self.logger.error( msgStr )

        try:
            self.settle()
        except Exception as e:
            self.logger.error( e )
            
        try:
            self.selOptions( snapDate )
        except Exception as e:            
            msgStr = e + '; Options selection was unsuccessful!'
            self.logger.error( msgStr )

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

    def buildMod( self, snapDate ):

        t0 = time.time()
        
        maxOosDt = snapDate
        maxTrnDt = maxOosDt - datetime.timedelta( minutes = 5 )
        minTrnDt = maxTrnDt - pd.DateOffset( years = self.nTrnYears )        

        tmpStr   = snapDate.strftime( '%Y-%m-%d_%H:%M:%S' )
        modFile  = self.modHead + tmpStr + '.dill'
        modFile  = os.path.join( self.modDir, modFile )

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

        sFlag = mfdMod.build()

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

        return True

    def settle( self ):
        pass
    
    def saveMod( self, mfdMod, modFile ):

        mfdMod.save( modFile )

        if not os.path.exists( modFile ):
            self.logger.error( 'The model file was not generated!' )

    def selOptions( self, snapDate ):

        t0 = time.time()

        self.logger.info( 'Building portfolio for snapdate %s', str( snapDate ) )        

        prtFile  = self.prtHead + tmpStr + '.json'
        prtFile  = os.path.join( self.prtDir, prtFile )
        
        minDate = snapDate + pd.DateOffset( days   = 1 )
        maxDate = snapDate + pd.DateOffset( months = self.maxMonths )
        
        assetHash   = self.getAssetHash()
        cash        = self.getCashValue()
        maxPriceC   = self.maxRatioC * cash
        maxPriceA   = self.maxRatioA * cash
        options     = self.getOptions()

        prtObj = MfdOptionsPrt( modFile     = modFile,
                                assetHash   = assetHash,
                                curDate     = snapDate,
                                minDate     = minDate,
                                maxDate     = maxDate,
                                maxPriceC   = maxPriceC,
                                maxPriceA   = maxPriceA,
                                minProb     = self.minProb,
                                maxCands    = None,                         
                                rfiDaily    = 0.0,
                                tradeFee    = self.tradeFee,
                                nDayTimes   = 1140,
                                logFileName = None,                    
                                verbose     = 1          )

        selHash = prtObj.selOptions( cash, options )

        self.savePrt( selHash, prtFile )
            
        self.logger.info( 'Building portfolio took %0.2f seconds!',
                          ( time.time() - t0 ) )

    def getAssetHash( self ):

        td = Tdam( refToken = REFRESH_TOKEN )
        
        assetHash = {}
        
        for symbol in self.etfs:
            assetHash[ symbol ] = td.getQuote( symbol )

        for symbol in self.stocks:
            assetHash[ symbol ] = td.getQuote( symbol )            

        for symbol in self.futures:
            val, date = utl.getYahooLastValue( symbol, 'futures' )
            assetHash[ symbol ] = val

        for symbol in self.indexes:
            val, date = utl.getYahooLastValue( symbol, 'index' )
            assetHash[ symbol ] = val
    
        return assetHash

    def getCashValue( self ):

        td = Tdam( refToken = REFRESH_TOKEN )

    def getOptions( self ):
 
        td = Tdam( refToken = REFRESH_TOKEN )
        
        options = []
        for symbol in self.assets:
            
            self.logger.info( 'Getting options for %s...', symbol )

            tmpList = td.getOptionsChain( symbol )

            options += tmpList
    
        self.logger.info( 'Found %d options contracts!', len( options ) )

        return options
    
    def savePrt( self, selHash, prtFile ):

        json.dump( selHash, open( prtFile, 'w' ) )

        if not os.path.exists( prtFile ):
            self.logger.error( 'The portfolio file was not generated!' )
    
    def sendPrtAlert( self, selHash ):

        assets = list( selHash.keys() )
        pars   = {}
        tmpStr = ''

        for asset in assets:
            count   = selHash[ asset ]
            tmpStr += '\n %10s: %d \n' % ( asset, count )

        pars[ 'Options' ] = tmpStr

        tempFile = open( USR_EMAIL_TEMPLATE, 'r' )
        tempStr  = tempFile.read()
        msgStr   = EmailTemplate( tempStr ).substitute( pars )

        tempFile.close()

        self.logger.critical( msgStr )
        
        self.logger.info( 'Selected options sent to email lists!' )

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

    def run( self ):

        os.environ[ 'TZ' ] = self.timeZone
        
        if not SCHED_FLAG:
            self.process()
        else:
            schedule.every().day.at( self.schedTime ).do( self.process )
            
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


