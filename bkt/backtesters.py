# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import re
import time
import datetime
import dill
import logging
import pickle
import numpy as np
import pandas as pd

from multiprocessing import Process, Pool

sys.path.append( os.path.abspath( '../' ) )

import utl.utils as utl

from mod.mfdMod import MfdMod
from prt.prt import MfdPrt

# ***********************************************************************
# Set some parameters 
# ***********************************************************************

SEND_ALERTS = True

DEV_LIST = [ 'babak.emami@gmail.com' ]

# ***********************************************************************
# Class MfdPrtBacktester: General backtester for mfd / prt
# ***********************************************************************

class MfdPrtBacktester():

    def __init__(   self,
                    velNames,
                    assets,                          
                    dfFile,
                    bktBegDate,
                    bktEndDate,
                    sType       = 'ETF',
                    maxAssets   = 5,
                    nEvalDays   = 30,
                    nTrnDays    = 360,
                    nOosDays    = 3,
                    nPrdDays    = 1,
                    nMadDays    = 30,
                    maxOptItrs  = 500,
                    optTol      = 5.0e-2,
                    regCoef     = 5.0e-3,
                    factor      = 4.0e-05,
                    outBktFile  = 'portfolio.json',
                    modFlag     = True,
                    modHead     = 'model_',
                    prtHead     = 'weights_',
                    modDir      = 'models',
                    prtDir      = 'models',
                    logFileName = None,
                    verbose     = 1           ):

        assert set( assets ).issubset( set( velNames ) ), \
            'Assets should be a subset of velNames!'

        assert maxAssets <= len( assets ), \
            'maxAssets should be <= number of assets in the pool!'

        if not os.path.exists( modDir ):
            os.makedirs( modDir )

        if not os.path.exists( prtDir ):
            os.makedirs( prtDir )
        
        self.velNames    = velNames
        self.assets      = assets
        self.maxAssets   = maxAssets
        self.dfFile      = dfFile
        self.bktBegDate  = pd.to_datetime( bktBegDate )
        self.bktEndDate  = pd.to_datetime( bktEndDate )
        self.sType       = sType
        self.nEvalDays   = nEvalDays
        self.nTrnDays    = nTrnDays
        self.nOosDays    = nOosDays
        self.nPrdDays    = nPrdDays
        self.nMadDays    = nMadDays        
        self.maxOptItrs  = maxOptItrs
        self.optTol      = optTol
        self.regCoef     = regCoef
        self.factor      = factor
        self.outBktFile  = outBktFile
        self.modFlag     = modFlag
        self.modHead     = modHead
        self.prtHead     = prtHead        
        self.modDir      = modDir
        self.prtDir      = prtDir
        self.logFileName = logFileName        
        self.verbose     = verbose
        self.logger      = utl.getLogger( logFileName, verbose )

        if SEND_ALERTS:
            devAlertHd = utl.getAlertHandler( alertLevel = logging.CRITICAL,
                                              subject    = 'Regarding the Mfd/Prt backtest!',
                                              mailList   = DEV_LIST )
            self.logger.addHandler( devAlertHd )
        
    def backtest( self ):

        snapDate = self.bktBegDate
        pool     = Pool()

        while snapDate <= self.bktEndDate:

            while True:
                if snapDate.isoweekday() not in [ 6, 7 ]:
                    break
                else:
                    snapDate += datetime.timedelta( days = 1 )

            pool.apply_async( self.build, args = ( snapDate, ) )

            snapDate = snapDate + datetime.timedelta( days = self.nPrdDays )

        pool.close()
        pool.join()

        self.combine()

        self.logger.critical( 'Backtest is done successfully!' )
        
    def build( self, snapDate ):

        snapDate = pd.to_datetime( snapDate )
        snapDate = snapDate.strftime( '%Y-%m-%d %H:%M:%S' )
        snapDate = pd.to_datetime( snapDate )

        self.logger.info( 'Processing snapDate %s ...' % str( snapDate ) )

        maxOosDt = snapDate
        maxTrnDt = maxOosDt - datetime.timedelta( days = self.nOosDays )
        minTrnDt = maxTrnDt - datetime.timedelta( days = self.nTrnDays )

        tmpStr   = snapDate.strftime( '%Y-%m-%d_%H:%M:%S' )
        modFile  = self.modHead + tmpStr + '.dill'
        prtFile  = self.prtHead + tmpStr + '.pkl'
        modFile  = os.path.join( self.modDir, modFile )
        prtFile  = os.path.join( self.prtDir, prtFile )

        t0       = time.time()

        if self.modFlag:
            
            mfdMod = MfdMod( dfFile       = self.dfFile,
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
                self.logger.critical( msgStr )

            if sFlag:
                self.logger.info( 'Building model took %0.2f seconds!',
                                  ( time.time() - t0 ) )
            else:
                self.logger.critical( 'The model did not converge!' )
                return False

            mfdMod.save( modFile )

        else:
            mfdMod = dill.load( open( modFile, 'rb' ) )

        if not os.path.exists( modFile ):
            self.logger.critical( 'New model file is not written to disk!' )
            return False
            
        self.logger.info( 'Building portfolio for snapdate %s', str( snapDate ) )

        t0     = time.time()
        wtHash = {}
        curDt  = snapDate
        endDt  = snapDate + datetime.timedelta( days = self.nPrdDays )

        nPrdTimes = int( self.nPrdDays * 19 * 60 )
        nRetTimes = int( self.nMadDays * 19 * 60 )

        if self.maxAssets is None:
            assets = self.assets
        else:
            try:
                eDf = utl.sortAssets( symbols = self.assets,
                                      nDays   = self.nEvalDays,
                                      sType   = self.sType,
                                      logger  = self.logger     )
                assets = list( eDf.asset )[:self.maxAssets]
            except Exception as e:
                self.logger.critical( e )
        
        mfdPrt = MfdPrt( modFile      = modFile,
                         assets       = assets,
                         nRetTimes    = nRetTimes,
                         nPrdTimes    = nPrdTimes,
                         strategy     = 'mad',
                         minProbLong  = 0.5,
                         minProbShort = 0.5,
                         vType        = 'vel',
                         fallBack     = 'macd',
                         logFileName  = None,
                         verbose      = 1          )

        try:
            dateKey = snapDate.strftime( '%Y-%m-%d' )
            wtHash[ dateKey ] = mfdPrt.getPortfolio()
        except Exception as e:
            msgStr = e + '; Portfolio build was unsuccessful!'
            self.logger.critical( msgStr )

        pickle.dump( wtHash, open( prtFile, 'wb' ) )    
            
        if not os.path.exists( prtFile ):
            self.logger.critical( 'New portfolio file is not written to disk!' )
            return False
        
        self.logger.info( 'Building portfolio took %0.2f seconds!',
                          ( time.time() - t0 ) )

        return True

    def combine( self ):
        
        pattern = self.prtHead + '\d+-\d+-\d+_\d+:\d+:\d+.pkl'        
        wtHash  = {}

        for fileName in os.lisdir( self.prtDir ):

            if not re.search( pattern, fileName ):
                continue

            filePath = os.path.join( self.prtDir, fileName )
            tmpHash = pickle.load( open( filePath, 'rb' ) )
            dateStr = list( tmpHash.keys() )[0]
            wtHash[ dateStr ] = tmpHash[ dateStr ]
            
        json.dump( wtHash, open( self.outBktFile, 'w' ) )

