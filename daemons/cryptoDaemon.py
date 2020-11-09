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
import numpy as np
import pandas as pd

from google.cloud import storage

from daemonBase import Daemon, EmailTemplate

sys.path.append( os.path.abspath( '../' ) )

import utl.utils as utl
import ptc.ptc as ptc

from dat.assets import INDEXES, CRYPTOS
from mod.mfdMod import MfdMod
from prt.prt import MfdPrt
from brk.rbin import Rbin

# ***********************************************************************
# Set some parameters 
# ***********************************************************************

ASSETS  = list( set( CRYPTOS ) - { 'ZEC' } )
INDEXES = INDEXES + [ 'VIX' ]

NUM_TRN_DAYS  = 360
NUM_OOS_DAYS  = 3
NUM_PRD_MINS  = 24 * 60
NUM_MAD_DAYS  = 30
MAX_OPT_ITRS  = 500
OPT_TOL       = 1.0e-5
REG_COEF      = 5.0e-3                    
FACTOR        = 4.0e-05

PTC_FLAG      = True
PTC_MIN_VIX   = None
PTC_MAX_VIX   = 60.0

GNP_FLAG      = True
GNP_STD_COEF  = 1.0
GNP_PERS_OFF  = 4
GNP_MIN_ROWS  = 14
RET_FILE      = '/var/crypto_returns/crypto_return_file.csv'

MOD_HEAD      = 'crypto_model_'
PRT_HEAD      = 'crypto_weights_'
PTC_HEAD      = 'ptc_'
MOD_DIR       = '/var/crypto_models'
PRT_DIR       = '/var/crypto_weights'
DAT_DIR       = '/var/crypto_data'
BASE_DAT_DIR  = '/var/data'
PTC_DIR       = '/var/pt_classifiers'
TIME_ZONE     = 'UTC'
SCHED_TIMES   = [ '21:00' ]
LOG_FILE_NAME = '/var/log/crypto_prt_builder.log'
VERBOSE       = 1

PID_FILE      = '/var/run/crypto_prt_builder.pid'

USR_EMAIL_TEMPLATE = '/home/babak/opti-trade/daemons/templates/user_portfolio_email.txt'

DEV_LIST = [ 'babak.emami@gmail.com' ]
USR_LIST = []

DEBUG_MODE = False

if DEBUG_MODE:
    SCHED_FLAG = False
    DRY_RUN    = True
else:
    SCHED_FLAG = True
    DRY_RUN    = False

NUM_DAYS_DATA_CLEAN = 2
NUM_DAYS_MOD_CLEAN  = 30
NUM_DAYS_PRT_CLEAN  = 730
NUM_DAYS_BUCKET_CLEAN = 5

GOOGLE_STORAGE_JSON = '/home/babak/opti-trade/daemons/keyfiles/google_storage.json'
GOOGLE_BUCKET = 'prt-storage'

# ***********************************************************************
# Class CryptoPrtBuilder: Daemon to build crypto portfolios
# ***********************************************************************

class CryptoPrtBuilder( Daemon ):

    def __init__(   self,
                    assets      = ASSETS,
                    indexes     = INDEXES,
                    nTrnDays    = NUM_TRN_DAYS,
                    nOosDays    = NUM_OOS_DAYS,
                    nPrdMinutes = NUM_PRD_MINS,
                    nMadDays    = NUM_MAD_DAYS,                                        
                    maxOptItrs  = MAX_OPT_ITRS,
                    optTol      = OPT_TOL,
                    regCoef     = REG_COEF,                    
                    factor      = FACTOR,
                    modHead     = MOD_HEAD,
                    prtHead     = PRT_HEAD,
                    ptcHead     = PTC_HEAD,
                    modDir      = MOD_DIR,
                    prtDir      = PRT_DIR,
                    datDir      = DAT_DIR,
                    baseDatDir  = BASE_DAT_DIR,
                    ptcDir      = PTC_DIR,                    
                    schedTimes  = SCHED_TIMES,
                    logFileName = LOG_FILE_NAME,
                    verbose     = VERBOSE         ):

        Daemon.__init__( self, PID_FILE )

        self.assets      = assets
        self.indexes     = indexes
        self.nTrnDays    = nTrnDays
        self.nOosDays    = nOosDays
        self.nPrdMinutes = nPrdMinutes
        self.nMadDays    = nMadDays                
        self.maxOptItrs  = maxOptItrs
        self.optTol      = optTol
        self.regCoef     = regCoef
        self.factor      = factor
        self.modHead     = modHead
        self.prtHead     = prtHead
        self.ptcHead     = ptcHead                
        self.modDir      = modDir
        self.prtDir      = prtDir
        self.datDir      = datDir
        self.baseDatDir  = baseDatDir
        self.ptcDir      = ptcDir
        self.schedTimes  = schedTimes
        self.logFileName = logFileName        
        self.verbose     = verbose
        self.velNames    = assets + indexes + [ 'ZEC' ]
        self.gnpNextDate = None
        
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
                                          subject    = 'A message from crypto trader!',
                                          mailList   = DEV_LIST )
        
        usrAlertHd = utl.getAlertHandler( alertLevel = logging.CRITICAL,
                                          subject    = 'A message from crypto trader!',
                                          mailList   = USR_LIST )

        self.logger.addHandler( devAlertHd )
        self.logger.addHandler( usrAlertHd )
        
        self.dfFile = None        

        self.logger.info( 'Daemon is initialized ...' )            

    def build( self ):

        os.environ[ 'TZ' ] = TIME_ZONE

        snapDate = datetime.datetime.now()
        snapDate = snapDate.strftime( '%Y-%m-%d %H:%M:%S' )
        snapDate = pd.to_datetime( snapDate )

        self.logger.info( 'Processing snapDate %s ...' % str( snapDate ) )

        try:
            self.setDfFile( snapDate )
        except Exception as e:
            self.logger.error( e )

        doGnpFlag = False
        try:
            doGnpFlag = self.doGainPreserve( snapDate )
        except Exception as e:
            self.logger.error( e )

        if doGnpFlag:
            
            self.logger.critical(
                'Gain preservation case! Trading abstinence!'
            )
            
            wtHash = {}
            for asset in self.assets:
                wtHash[ asset ] = 0.0
                
            try:
                self.trade( wtHash )
            except Exception as e:
                msgStr = e + '; Trade was not successful!'
                self.logger.error( msgStr )

            return True
        else:
            self.logger.info(
                'No gain preservation detected! Continue with trading!'
            )

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

        nPrdTimes = self.nPrdMinutes
        nRetTimes = int( self.nMadDays * 19 * 60 )

        quoteHash = self.getQuoteHash( snapDate, mfdMod )
        
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
            
        try:
            self.buildPTC( self.assets, snapDate )
        except Exception as e:
            msgStr = e + '; PTC build was unsuccessful!'
            self.logger.error( msgStr )

        try:
            wtHash = self.adjustPTC( wtHash, snapDate )
        except Exception as e:
            msgStr = e + '; PTC adjustment was unsuccessful!'
            self.logger.error( msgStr )            

        try:
            wtHash = self.adjustNoShort( wtHash )
        except Exception as e:
            msgStr = e + '; No short adjustment was unsuccessful!'
            self.logger.error( msgStr )            
            
        self.savePrt( wtHash, prtFile )
            
        if not os.path.exists( prtFile ):
            self.logger.error( 'New portfolio file is not written to disk!' )
            return False
        
        self.logger.info( 'Building portfolio took %0.2f seconds!',
                          ( time.time() - t0 ) )

        try:
            self.trade( wtHash )
        except Exception as e:
            msgStr = e + '; Trade was not successful!'
            self.logger.error( msgStr )        

        try:
            self.sendPrtAlert( wtHash )
        except Exception as e:
            msgStr = e + '; Portfolio alert was NOT sent!'
            self.logger.error( msgStr )

        self.clean( self.datDir, NUM_DAYS_DATA_CLEAN )
        self.clean( self.modDir, NUM_DAYS_MOD_CLEAN  )
        self.clean( self.prtDir, NUM_DAYS_PRT_CLEAN  )

        try:
            self.cleanBucket( NUM_DAYS_BUCKET_CLEAN )
        except Exception as e:
            self.logger.error(e)
            
        return True
        
    def setDfFile( self, snapDate ):

        self.logger.info( 'Getting data...' )
        
        nDays   = self.nTrnDays + self.nOosDays
        maxDate = pd.to_datetime( snapDate )
        minDate = maxDate - datetime.timedelta( days = nDays )

        self.logger.info( 'Reading available data...' )

        indexDf = utl.mergeSymbols( symbols = self.indexes,
                                    datDir  = self.baseDatDir,
                                    fileExt = 'pkl',
                                    minDate = minDate,
                                    logger  = self.logger   )

        indexDf.Date = indexDf.Date.dt.tz_localize( 'America/New_York' )
        indexDf.Date = indexDf.Date.dt.tz_convert( 'UTC' )
        indexDf.Date = indexDf.Date.dt.tz_convert( None )

        oldDf = utl.mergeSymbols( symbols = self.assets,
                                  datDir  = self.baseDatDir,
                                  fileExt = 'pkl',
                                  minDate = minDate,
                                  logger  = self.logger     )

        oldDf = oldDf.merge( indexDf, how = 'left', on = 'Date' )
        oldDf = oldDf.interpolate( method = 'linear' )
        oldDf = oldDf.dropna()
        oldDf = oldDf.sort_values( 'Date' )
        oldDf = oldDf.reset_index( drop = True )
        
        self.logger.info( 'Getting new data...' )

        try:
            indexDf = utl.getYahooData(
                indexes = self.indexes,
                nDays   = 5,
                logger  = self.logger
            )
            indexDf.Date = indexDf.Date.dt.tz_localize( 'America/New_York' )
            indexDf.Date = indexDf.Date.dt.tz_convert( 'UTC' )
            indexDf.Date = indexDf.Date.dt.tz_convert( None )
            
            newDf = utl.getCryptoCompareData(
                self.assets,
                logger  = self.logger
            )

            newDf = newDf.merge( indexDf, how = 'left', on = 'Date' )
            newDf = newDf.interpolate( method = 'linear' )
            newDf = newDf.dropna()
            newDf = newDf.sort_values( 'Date' )
            newDf = newDf.reset_index( drop = True )

            self.logger.info( 'Done with getting new data!' )

            self.logger.info( 'Merging old and new data...' )
        
            newDf = newDf[ newDf.Date > oldDf.Date.max() ]
            newDf = pd.concat( [ oldDf, newDf ] )
        except Exception as e:
            msgStr = '%s; Could not get new data, old data max date is %s!' \
                     % (e, str( oldDf.Date.max() ))
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

        items = [ 'Date' ] + self.assets + self.indexes

        for item in items:
            if not item in df.columns:
                msgStr =' Urgent: The file %s does not have a %s column! Stopping the daemon...' %\
                    ( self.dfFile, item )
                self.logger.error( msgStr )
                self.stop()

        self.logger.info( 'The new data file looks ok!' )

    def doGainPreserve( self, snapDate ):
        
        if not GNP_FLAG:
            return False

        self.logger.info( 'Evaluate return from previous trade...' )

        prevDate = snapDate - \
            datetime.timedelta( minutes = self.nPrdMinutes )

        assert self.dfFile is not None, 'The data file is not set yet!'
        
        df = pd.read_pickle( self.dfFile )
        df = df[ df.Date >= prevDate ]
        
        rbin = Rbin(
            os.getenv( "RBIN_USERNAME" ),
            os.getenv( "RBIN_PASSKEY" )
        )
        
        qtyHash = rbin.getPortfolio()
        totVal  = rbin.getTotalValue()
        totCash = rbin.getCashBalance()
        
        prevVal = 0.0        
        currVal = 0.0
        for symbol in qtyHash:
            qty = float( qtyHash[ symbol ] )
            
            prevPrice = list( df[ symbol ] )[0]            
            currPrice = list( df[ symbol ] )[-1]
            prevVal  += qty * prevPrice
            currVal  += qty * currPrice            

        assert prevVal >= 0, 'Previous value cannot be negative!'
        assert currVal >= 0, 'Current value cannot be negative!'
        
        fct = prevVal
        if fct != 0:
            fct = 1.0 / fct

        retVal = fct * ( currVal - prevVal )

        self.logger.info(
            'Return compared to previous trade is %0.6f %%...',
            100.0 * retVal
        )
        
        newRetDf = pd.DataFrame(
            {
                'Date': [ snapDate, ],
                'Return': [ retVal ],
                'Asset Balance': [ currVal ],                
                'Tot Account Val': [ totVal ],
                'Tot Account Cash': [ totCash ],
                'Source': [ 'Actual' ],
            }
        )
                
        if os.path.exists( RET_FILE ):
            retDf = pd.read_csv( RET_FILE )
            retDf = pd.concat( [ retDf, newRetDf ] ) 
        else:
            retDf = newRetDf

        retDf.to_csv( RET_FILE, index = False )

        doGnpFlag = False
        
        if self.gnpNextDate is not None and \
           snapDate < self.gnpNextDate:
            doGnpFlag = True
        elif retDf.shape[0] < GNP_MIN_ROWS:
            doGnpFlag = False
        else:
            retMean = retDf.Return.mean()
            retStd  = retDf.Return.std()
            tmpVal  = retMean + GNP_STD_COEF * retStd
            
            if retVal > tmpVal:
                doGnpFlag = True
                self.gnpNextDate = snapDate + \
                    datetime.timedelta(
                        minutes = GNP_PERS_OFF * self.nPrdMinutes
                    )            
            else:
                doGnpFlag = False

        return doGnpFlag
    
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

    def getQuoteHash( self, snapDate, mfdMod ):

        self.logger.info( 'Getting quotes...' )
        
        quoteHash = {}
        
        ecoMfd = mfdMod.ecoMfd

        for m in range( ecoMfd.nDims ):

            asset = ecoMfd.velNames[m]

            if asset not in self.assets:
                continue
        
            tmp       = ecoMfd.deNormHash[ asset ]
            slope     = tmp[0]
            intercept = tmp[1]
        
            quoteHash[ asset ] = \
                slope * ecoMfd.actOosSol[m][-1] + intercept
            
        return quoteHash

    def buildPTC( self, symbols, snapDate ):

        if not PTC_FLAG:
            return

        minTrnDate = snapDate - datetime.timedelta( days = self.nTrnDays )
        
        for symbol in symbols:

            symFile = os.path.join( self.baseDatDir,
                                    '%s.pkl' % symbol )
            vixFile = os.path.join( self.baseDatDir,
                                    'VIX.pkl' )            

            ptcObj  = ptc.PTClassifier( symbol      = symbol,
                                        symFile     = symFile,
                                        vixFile     = vixFile,
                                        ptThreshold = 1.0e-2,
                                        nPTAvgDays  = None,
                                        testRatio   = 0,
                                        method      = 'bayes',
                                        minVix      = PTC_MIN_VIX,
                                        maxVix      = PTC_MAX_VIX,
                                        minTrnDate  = minTrnDate,
                                        logFileName = None,                    
                                        verbose     = 1          )

            ptcObj.classify()

            ptcFile = os.path.join( self.ptcDir,
                                    self.ptcHead + symbol + '.pkl' )
            
            self.logger.info( 'Saving the classifier to %s', ptcFile )
            
            ptcObj.save( ptcFile )

    def adjustPTC( self, wtHash, snapDate ):

        if not PTC_FLAG:
            return wtHash

        self.logger.info( 'Applying peak/trough classifiers to portfolio!' )
        
        dayDf = pd.read_pickle( self.dfFile )        

        dayDf[ 'Date' ] = dayDf.Date.astype( 'datetime64[ns]' )
    
        minDate = snapDate - \
            pd.DateOffset( days = 7 )
    
        dayDf = dayDf[ ( dayDf.Date >= minDate ) &
                       ( dayDf.Date <= snapDate ) ]
    
        dayDf[ 'Date' ] = dayDf.Date.\
            apply( lambda x : x.strftime( '%Y-%m-%d' ) )
    
        dayDf = dayDf.groupby( 'Date', as_index = False ).mean()

        dayDf[ 'Date' ] = dayDf.Date.astype( 'datetime64[ns]' )
        
        dayDf = dayDf.sort_values( [ 'Date' ], ascending = True )

        vixVal = list( dayDf.VIX )[-1]

        if PTC_MIN_VIX is not None and vixVal < PTC_MIN_VIX:
            self.logger.critical( 'Did not use PTC as current VIX of '
                                  '%0.2f is not in range!',
                                  vixVal )
            return wtHash

        if PTC_MAX_VIX is not None and vixVal > PTC_MAX_VIX:
            self.logger.critical( 'Did not use PTC as current VIX of '
                                  '%0.2f is not in range!',
                                  vixVal )
            return wtHash

        for symbol in wtHash:
        
            dayDf[ 'vel' ] = np.gradient( dayDf[ symbol ], 2 )
            dayDf[ 'acl' ] = np.gradient( dayDf[ 'vel' ], 2 )

            symVal = list( dayDf.acl )[-1] 
            
            ptcFile = os.path.join( self.ptcDir,
                                    self.ptcHead + symbol + '.pkl' )
            
            obj = pickle.load( open( ptcFile, 'rb' ) )

            X = np.array( [ [ symVal ] ] )
        
            ptTag = obj.predict( X )[0]

            if ptTag == ptc.PEAK:
                
                self.logger.critical( 'A peak is detected for %s!', symbol )
                
                if wtHash[ symbol ] > 0:
                    self.logger.critical( 'Changing weight for %s from %0.2f to '
                                          '%0.2f as a peak was detected!',
                                          symbol,
                                          wtHash[ symbol ],
                                          -wtHash[ symbol ] )
                    
                    wtHash[ symbol ] = -wtHash[ symbol ]
                    
            elif ptTag == ptc.TROUGH:

                self.logger.critical( 'A trough is detected for %s!',
                                      symbol   )
            
                if wtHash[ symbol ] < 0:
                    self.logger.critical( 'Changing weight for %s from %0.2f to '
                                          '%0.2f as a trough was detected!',
                                          symbol,
                                          wtHash[ symbol ],
                                          -wtHash[ symbol ] )
                    
                    wtHash[ symbol ] = -wtHash[ symbol ]

        return wtHash

    def adjustNoShort( self, wtHash ):

        for symbol in wtHash:
            wtHash[ symbol ] = max( 0.0, wtHash[ symbol ] )
        
        sumAbs = sum( [ abs( x ) for x in wtHash.values() ] )
        
        sumAbsInv = 1.0
        if sumAbs > 0:
            sumAbsInv = 1.0 / sumAbs

        for symbol in wtHash:
            wtHash[ symbol ] = sumAbsInv * wtHash[ symbol ]

        return wtHash

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

    def trade( self, wtHash ):

        os.environ[ 'TZ' ] = TIME_ZONE
        
        if not DRY_RUN:
            
            try:
                rbin = Rbin(
                    os.getenv( "RBIN_USERNAME" ),
                    os.getenv( "RBIN_PASSKEY" )
                )
                self.logger.info( 'Connected to Robinhood!' )
            except Exception as e:
                self.logger.error( e )
                
            self.logger.info( 'Starting to trade on Robinhood...' )

            try:
                rbin.adjWeights( wtHash = wtHash )
            except Exception as e:
                logging.error( 'Trading failed: %s!' % e )

            self.logger.info( 'Done with trading!' )            
                
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
        
        tzObj    = pytz.timezone( TIME_ZONE )
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

        os.environ[ 'TZ' ] = TIME_ZONE
        
        if not SCHED_FLAG:
            self.build()
        else:
            for item in self.schedTimes:
                schedule.every().day.at( item ).do( self.build )
            
            while True: 
                schedule.run_pending() 
                time.sleep( 60 )
              
# ***********************************************************************
# Run daemon
# ***********************************************************************

if __name__ ==  '__main__':

    daemon = CryptoPrtBuilder()
    
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


