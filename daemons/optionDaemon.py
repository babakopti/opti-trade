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

sys.path.append( os.path.abspath( '../' ) )

import utl.utils as utl

from daemonBase import Daemon, EmailTemplate
from dat.assets import OPTION_ETFS as ETFS
from dat.assets import FUTURES, INDEXES
from mod.mfdMod import MfdMod
from prt.prt import MfdOptionsPrt
from brk.tdam import Tdam

# ***********************************************************************
# Set some parameters 
# ***********************************************************************

INDEXES  = list( set( INDEXES ) - { 'TRAN', 'TYX' } )
ETFS     = list( set( ETFS ) - { 'OIH' } )
STOCKS   = []

ASSETS   = ETFS

NUM_TRN_YEARS = 5
NUM_OOS_MINS  = 5
STEP_SIZE     = None
MAX_OPT_ITRS  = 500
OPT_TOL       = 1.0e-3
REG_COEF      = 1.0e-3                    
FACTOR        = 1.0e-5

NUM_WEEKDAYS_BUY    = 0
MAX_OPTION_MONTHS   = 3
MAX_PRICE_CONTRACT  = 500.0
MAX_PRICE_ASSET     = 500.0
MAX_HOLDING_ASSET   = 1000.0
MAX_RATIO_EXPOSURE  = 1.0
MAX_SELECTION_COUNT = 1
MIN_PROBABILITY     = 0.496
OPTION_TRADE_FEE    = 0.75
OPTION_TRADE_TYPE   = None

MOD_HEAD      = 'option_model_'
PRT_HEAD      = 'option_prt_'
CHAIN_HEAD    = 'option_chain_'
MOD_DIR       = '/var/option_models'
PRT_DIR       = '/var/option_prt'
DAT_DIR       = '/var/option_data'
BASE_DAT_DIR  = '/var/data'
PI_DAT_DIR    = '/var/pi_data'
CHAIN_DIR     = '/var/option_chains'
TIME_ZONE     = 'America/New_York'
SCHED_TIME    = '15:00'
LOG_FILE_NAME = '/var/log/option_prt_builder.log'
VERBOSE       = 1

PID_FILE      = '/var/run/option_prt_builder.pid'

USR_EMAIL_TEMPLATE = '/home/babak/opti-trade/daemons/templates/user_portfolio_email_option.txt'

DEV_LIST = [ 'babak.emami@gmail.com' ]
USR_LIST = []

TOKEN_FILE = '../brk/tokens/refresh_token_pany_2020-12-17.txt'

with open( TOKEN_FILE, 'r' ) as fHd:
    REFRESH_TOKEN = fHd.read()[:-1]

OPTION_ACCOUNT_ID = '868894929'

GOOGLE_STORAGE_JSON = '/home/babak/opti-trade/daemons/keyfiles/google_storage.json'
GOOGLE_BUCKET = 'prt-storage'
GOOGLE_PREFIX = 'options-chains'

DEBUG_MODE = False

if DEBUG_MODE:
    SCHED_FLAG = False
    DRY_RUN    = True
else:
    SCHED_FLAG = True
    DRY_RUN    = False

NUM_DAYS_DATA_CLEAN = 3
NUM_DAYS_MOD_CLEAN  = 3
NUM_DAYS_PRT_CLEAN  = 90

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
                    nOosMinutes = NUM_OOS_MINS,
                    stepSize    = STEP_SIZE,
                    maxOptItrs  = MAX_OPT_ITRS,
                    optTol      = OPT_TOL,
                    regCoef     = REG_COEF,                    
                    factor      = FACTOR,
                    maxMonths   = MAX_OPTION_MONTHS,
                    maxPriceC   = MAX_PRICE_CONTRACT,
                    maxPriceA   = MAX_PRICE_ASSET,
                    maxHoldA    = MAX_HOLDING_ASSET,
                    maxRatioExp = MAX_RATIO_EXPOSURE,
                    maxSelCnt   = MAX_SELECTION_COUNT,
                    minProb     = MIN_PROBABILITY,
                    tradeFee    = OPTION_TRADE_FEE,
                    optionType  = OPTION_TRADE_TYPE,
                    modHead     = MOD_HEAD,
                    prtHead     = PRT_HEAD,
                    chainHead   = CHAIN_HEAD,
                    modDir      = MOD_DIR,
                    prtDir      = PRT_DIR,
                    datDir      = DAT_DIR,
                    baseDatDir  = BASE_DAT_DIR,
                    piDatDir    = PI_DAT_DIR,
                    chainDir    = CHAIN_DIR,
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
        self.nOosMinutes = nOosMinutes
        self.stepSize    = stepSize
        self.maxOptItrs  = maxOptItrs
        self.optTol      = optTol
        self.regCoef     = regCoef
        self.factor      = factor
        self.maxMonths   = maxMonths
        self.maxPriceC   = maxPriceC
        self.maxPriceA   = maxPriceA
        self.maxHoldA    = maxHoldA        
        self.maxRatioExp = maxRatioExp
        self.maxSelCnt   = maxSelCnt
        self.minProb     = minProb
        self.tradeFee    = tradeFee
        self.optionType  = optionType
        self.modHead     = modHead
        self.prtHead     = prtHead
        self.chainHead   = chainHead
        self.modDir      = modDir
        self.prtDir      = prtDir
        self.datDir      = datDir
        self.baseDatDir  = baseDatDir
        self.piDatDir    = piDatDir
        self.chainDir    = chainDir
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
        
        self.dfFile   = None
        self.modFile  = None
        self.prtObj   = None
        self.alertStr = ''

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

            holidays = pmc.get_calendar('NYSE').holidays().holidays

            if pd.to_datetime( snapDate.strftime( '%Y-%m-%d' ) ) in holidays:            
                self.logger.critical( 'Not running as today is a NYSE holiday!' )
                return
        
        self.logger.info( 'Processing snapDate %s ...' % str( snapDate ) )

        try:
            self.setDfFile( snapDate )
        except Exception as err:
            self.logger.error( err )

        try:
            self.buildMod( snapDate )
        except Exception as err:            
            self.logger.error( err )

        try:
            self.setPrtObj( snapDate )
        except Exception as err:
            self.logger.error( err )
            
        try:
            self.settle()
        except Exception as err:
            self.logger.error( err )
            
        try:
            self.selTradeOptions( snapDate )
        except Exception as err:            
            self.logger.error( err )

        try:
            self.sendPrtAlert()
        except Exception as err:
            self.logger.error( err )

        self.clean( self.datDir, NUM_DAYS_DATA_CLEAN )
        self.clean( self.modDir, NUM_DAYS_MOD_CLEAN  )
        self.clean( self.prtDir, NUM_DAYS_PRT_CLEAN  )

        return True

    def setDfFile( self, snapDate ):

        t0 = time.time()
        
        maxDate  = pd.to_datetime( snapDate )
        minDate  = maxDate - \
                   datetime.timedelta( minutes = self.nOosMinutes ) - \
                   pd.DateOffset( years = self.nTrnYears )        

        symbols = self.etfs + self.stocks + self.futures + self.indexes

        self.logger.info( 'Getting data for %d symbols...', len( symbols ) )
        
        self.logger.info( 'Reading pitrading data...' )

        piDf = utl.mergePiSymbols( symbols = symbols,
                                   datDir  = self.piDatDir,
                                   minDate = minDate,
                                   logger  = self.logger )
        
        self.logger.info( 'Reading newer available data...' )
        
        oldDf = utl.mergeSymbols( symbols = symbols,
                                  datDir  = self.baseDatDir,
                                  fileExt = 'pkl',
                                  minDate = minDate,
                                  logger  = self.logger )
 
        piDf  = piDf[ piDf.Date < oldDf.Date.min() ]
        oldDf = pd.concat( [ piDf, oldDf ] )        
        
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
            
        self.logger.info( 'Getting data took %0.2f seconds!',
                          ( time.time() - t0 ) )
    
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

        tmpDf = pd.read_pickle( self.dfFile )

        maxOosDt = min( tmpDf.Date.max(), snapDate )
        maxTrnDt = maxOosDt - datetime.timedelta( minutes = self.nOosMinutes )
        minTrnDt = maxTrnDt - pd.DateOffset( years = self.nTrnYears )        

        tmpStr   = snapDate.strftime( '%Y-%m-%d_%H:%M:%S' )
        modFile  = self.modHead + tmpStr + '.dill'
        modFile  = os.path.join( self.modDir, modFile )

        mfdMod   = MfdMod( dfFile       = self.dfFile,
                           minTrnDate   = minTrnDt,
                           maxTrnDate   = maxTrnDt,
                           maxOosDate   = maxOosDt,
                           velNames     = self.velNames,
                           stepSize     = self.stepSize,
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

        self.modFile = modFile

        return True

    def setPrtObj( self, snapDate ):

        self.logger.info( 'Setting portfolio object...' )

        minDate = snapDate + pd.DateOffset( days   = 1 )
        maxDate = snapDate + pd.DateOffset( months = self.maxMonths )

        self.logger.info( 'Setting assetHash...' )
        
        assetHash = self.getAssetHash()
        
        self.logger.info( 'Done with setting assetHash...' )
        
        self.prtObj = MfdOptionsPrt( modFile     = self.modFile,
                                     assetHash   = assetHash,
                                     curDate     = snapDate,
                                     minDate     = minDate,
                                     maxDate     = maxDate,
                                     minProb     = self.minProb,
                                     rfiDaily    = 0.0,
                                     tradeFee    = self.tradeFee,
                                     nDayTimes   = 1140,
                                     logFileName = None,                    
                                     verbose     = 1          )

        self.logger.info( 'The prt object was set!' )

    def settle( self ):

        self.logger.info( 'Settling the current options holdings...' )

        try:
            td = Tdam( refToken = REFRESH_TOKEN, accountId = OPTION_ACCOUNT_ID )
        except Exception as e:
            self.logger.error( e )
        
        positions = td.getPositions()

        pattern = '(\w+)\_(\d+)(C|P)([\W|\d]+)'

        for position in positions:

            sType    = position[ 'instrument' ][ 'assetType' ]
            longQty  = position[ 'longQuantity' ]
            shortQty = position[ 'shortQuantity' ]

            if sType != 'OPTION':
                continue

            if longQty == 0 or shortQty > 0:
                continue

            symbol      = position[ 'instrument' ][ 'symbol' ]            
            assetSymbol = position[ 'instrument' ][ 'underlyingSymbol' ]
            oType       = position[ 'instrument' ][ 'putCall' ].lower()
            
            tmpTuple = re.findall( pattern, symbol )[0]

            if len( tmpTuple ) != 4 or\
               tmpTuple[0] != assetSymbol or\
               ( oType == 'call' and tmpTuple[2] != 'C' ) or\
               ( oType == 'put' and tmpTuple[2] != 'P' ):
                self.logger.error( 'Incorrect option symbol %s!', symbol )
                
            exprDate = pd.to_datetime( tmpTuple[1], format = '%m%d%y' )
            strike   = float( tmpTuple[3] )
            
            unitPrice = td.getQuote( symbol, 'bid' )

            option = { 'optionSymbol' : symbol,
                       'assetSymbol'  : assetSymbol,
                       'strike'       : strike,
                       'expiration'   : exprDate,
                       'contractCnt'  : 100,
                       'unitPrice'    : unitPrice,
                       'type'         : oType      }

            self.logger.info( 'Evaluating option %s...' % symbol )
            
            ( decision, prob ) = self.prtObj.getCurAction( option,
                                                           unitPrice )

            msgStr = 'Decision for %s is %s; Success probability is %0.3f' % \
                     ( symbol, decision, prob )

            self.alertStr += msgStr + '\n'
            
            self.logger.info( msgStr )
            
            if decision == 'sell_now':

                msgStr = 'Selling %d of %s...' % ( longQty, symbol )

                self.alertStr += msgStr + '\n'
                
                self.logger.info( msgStr )

                if not DRY_RUN:
                    td.order( symbol    = symbol,
                              quantity  = longQty,
                              sType     = 'OPTION',
                              action    = 'SELL_TO_CLOSE' )
                
            elif decision == 'exec_now':

                msgStr = 'Exercising %d of %s...' % ( longQty, symbol )
                
                self.alertStr += msgStr + '\n'
                
                self.logger.info( msgStr )
                
                if not DRY_RUN:                
                    td.order( symbol    = symbol,
                              orderType = 'EXERCISE',
                              quantity  = longQty,
                              sType     = 'OPTION',
                              action    = 'SELL_TO_CLOSE' )
                
            else:
                pass
            
    def saveMod( self, mfdMod, modFile ):

        mfdMod.save( modFile )

        if not os.path.exists( modFile ):
            self.logger.error( 'The model file was not generated!' )

    def selTradeOptions( self, snapDate ):

        self.logger.info( 'Selecting options for snapdate %s', str( snapDate ) )        

        tmpStr   = snapDate.strftime( '%Y-%m-%d_%H:%M:%S' )        
        prtFile  = self.prtHead + tmpStr + '.json'
        prtFile  = os.path.join( self.prtDir, prtFile )
        
        minDate = snapDate + pd.DateOffset( days   = 1 )
        maxDate = snapDate + pd.DateOffset( months = self.maxMonths )
        
        cash = self.getCashValue()

        exposedCash = self.maxRatioExp * cash

        self.logger.info( 'Amount of available cash is %0.2f; exposure is %0.2f!',
                          cash,
                          exposedCash )

        options = self.getOptions()

        selHash = self.prtObj.selOptions( options    = options,
                                          cash       = exposedCash,
                                          maxPriceC  = self.maxPriceC,
                                          maxPriceA  = self.maxPriceA,
                                          maxSelCnt  = self.maxSelCnt,
                                          optionType = self.optionType )

        self.savePrt( selHash, prtFile )

        self.trade( selHash )

    def getAssetHash( self ):

        try:
            td = Tdam( refToken = REFRESH_TOKEN, accountId = OPTION_ACCOUNT_ID )
        except Exception as e:
            self.logger.error( e )
        
        assetHash = {}
        
        for symbol in self.etfs:
            assetHash[ symbol ] = td.getQuote( symbol, 'last' )

        for symbol in self.stocks:
            assetHash[ symbol ] = td.getQuote( symbol, 'last' )            

        for symbol in self.futures:
            val, date = utl.getYahooLastValue( symbol,
                                               'futures',
                                               logger = self.logger )
            assetHash[ symbol ] = val

        for symbol in self.indexes:
            val, date = utl.getYahooLastValue( symbol,
                                               'index',
                                               logger = self.logger )
            assetHash[ symbol ] = val
    
        return assetHash

    def getCashValue( self ):

        try:
            td = Tdam( refToken = REFRESH_TOKEN, accountId = OPTION_ACCOUNT_ID )
        except Exception as e:
            self.logger.error( e )

        cash = td.getCashBalance()

        return cash

    def getOptions( self ):

        try:
            td = Tdam( refToken = REFRESH_TOKEN, accountId = OPTION_ACCOUNT_ID )
        except Exception as e:
            self.logger.error( e )
            
        options = []
        for symbol in self.assets:
            
            self.logger.info( 'Getting options for %s...', symbol )

            tmpList = td.getOptionsChain( symbol )

            options += tmpList
    
        self.logger.info( 'Found %d options contracts!', len( options ) )

        try:
            self.saveOptions( options )
        except Exception as e:
            self.logger.error( 'Unable to save option chains: %s', e )
            
        self.logger.info( 'Saved option chains for future use.' )

        filtOptions = self.filterMaxHolding( options )

        self.logger.info( 'Filtered out %d options to conform with max asset '
                          'holding constraint of %2.f.',
                          len( options ) - len( filtOptions ),
                          self.maxHoldA  )
        return filtOptions

    def saveOptions( self, options ):

        if self.prtObj is None:
            self.logger.warning( 'prtObj should be set before '
                                 'calling saveOptions!' )
            return

        self.logger.info( 'Saving options chains...' )
        
        curDate   = datetime.datetime.now()
        tmpStr    = curDate.strftime( '%Y-%m' )
        chainFile = self.chainHead + tmpStr + '.pkl'
        chainFile = os.path.join( self.chainDir, chainFile )
        
        if os.path.exists( chainFile ):
            oldDf = pd.read_pickle( chainFile )
        else:
            oldDf = pd.DataFrame()
            
        newHash = defaultdict( list )

        for option in options:
            
            exprDate = pd.to_datetime( option[ 'expiration' ] )
            
            if exprDate <= self.prtObj.curDate:
                continue
            
            if exprDate > self.prtObj.maxDate:
                continue
            
            option[ 'Prob' ] = self.prtObj.getProb( option )
            option[ 'DataDate' ] = curDate.strftime( '%Y-%m-%d' )
            
            for col in option:
                newHash[ col ].append( option[ col ] )

        newDf = pd.DataFrame( newHash )

        newDf = pd.concat( [ oldDf, newDf ] )

        newDf.to_pickle( chainFile )

        self.logger.info( 'Options chains saved to %s', chainFile )        

        client   = storage.Client.from_service_account_json( GOOGLE_STORAGE_JSON )
        bucket   = client.get_bucket( GOOGLE_BUCKET )
        baseName = os.path.basename( chainFile )
        tmpName  = GOOGLE_PREFIX + '/' + baseName
        blob     = bucket.blob( tmpName )
            
        with open( chainFile, 'rb' ) as fHd:
            blob.upload_from_file( fHd )

        self.logger.info( '%s was saved to bucket!', tmpName )        

    def filterMaxHolding( self, options ):

        try:
            td = Tdam( refToken = REFRESH_TOKEN, accountId = OPTION_ACCOUNT_ID )
        except Exception as e:
            self.logger.error( e )
        
        positions = td.getPositions()
        holdHash  = defaultdict( float )

        for position in positions:

            sType    = position[ 'instrument' ][ 'assetType' ]
            longQty  = position[ 'longQuantity' ]
            shortQty = position[ 'shortQuantity' ]

            if sType != 'OPTION':
                continue

            if longQty == 0 or shortQty > 0:
                continue

            symbol      = position[ 'instrument' ][ 'symbol' ]                        
            assetSymbol = position[ 'instrument' ][ 'underlyingSymbol' ]
            unitPrice   = td.getQuote( symbol, 'bid' )
            
            holdHash[ assetSymbol ] += 100 * unitPrice

        filtOptions = []

        for option in options:

            assetSymbol = option[ 'assetSymbol' ]            
            oCnt        = option[ 'contractCnt' ]
            unitPrice   = option[ 'unitPrice' ]
            price       = oCnt * unitPrice

            if price + holdHash[ assetSymbol ] > self.maxHoldA:
                continue

            filtOptions.append( option )
        
        return filtOptions
        
    def savePrt( self, selHash, prtFile ):

        json.dump( selHash, open( prtFile, 'w' ) )

        if not os.path.exists( prtFile ):
            self.logger.error( 'The portfolio file was not generated!' )

    def trade( self, selHash ):

        td = Tdam( refToken = REFRESH_TOKEN, accountId = OPTION_ACCOUNT_ID )
        
        self.alertStr += '\nSelected %d options contract(s)!\n\n' \
            % ( len( selHash.keys() ) )
                                                                     
        for symbol in selHash:

            quantity = int( selHash[ symbol ] )
            
            msgStr = 'Buying %d of %s...' % ( quantity, symbol )

            self.alertStr += msgStr + '\n'
            
            self.logger.info( msgStr )

            curDayInt = datetime.datetime.now().isoweekday()
            randDays  = np.random.randint( 1, 6, NUM_WEEKDAYS_BUY )
        
            self.logger.info( 'Randomly selected weekday(s) are %s, today is %d!',
                              str( randDays ),
                              curDayInt )
            
            if not DRY_RUN:
                if curDayInt in randDays:            
                    td.order( symbol    = symbol,
                              quantity  = quantity,
                              sType     = 'OPTION',
                              action    = 'BUY_TO_OPEN' )
                    
                    self.alertStr += 'Bought %d of %s...\n' % ( quantity, symbol )
                else:
                    msgStr = 'Not trading today!'
                    
                    self.alertStr += msgStr + '\n'
                    
                    self.logger.info( msgStr )
    
    def sendPrtAlert( self ):

        pars   = {}

        pars[ 'Options' ] = self.alertStr

        tempFile = open( USR_EMAIL_TEMPLATE, 'r' )
        tempStr  = tempFile.read()
        msgStr   = EmailTemplate( tempStr ).substitute( pars )

        tempFile.close()

        self.logger.critical( msgStr )
        
        self.logger.info( 'Traded options info sent to email lists!' )

        self.alertStr = ''

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

    daemon = OptionPrtBuilder()

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


