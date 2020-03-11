# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import logging
import schedule
import pandas as pd

from daemonBase import Daemon

sys.path.append( os.path.abspath( '../' ) )

import utl.utils as utl

from dat.assets import ETF_HASH, SUB_ETF_HASH, NEW_ETF_HASH, POP_ETF_HASH
from dat.assets import OPTION_ETFS, PI_ETFS
from dat.assets import INDEXES, PI_INDEXES
from dat.assets import FUTURES

# ***********************************************************************
# Set some parameters 
# ***********************************************************************

ETFS = list( ETF_HASH.keys() ) + list( ETF_HASH.values() ) +\
       list( SUB_ETF_HASH.keys() ) + list( SUB_ETF_HASH.values() ) +\
       list( NEW_ETF_HASH.keys() ) + list( NEW_ETF_HASH.values() ) +\
       list( POP_ETF_HASH.keys() ) + list( POP_ETF_HASH.values() )
ETFS = list( set( ETFS ) )

STOCKS = []

INDEXES = INDEXES + PI_INDEXES
INDEXES = list( set( INDEXES ) )

NUM_DAYS      = 3000
SOURCE        = 'kibot'
DAT_DIR       = '/var/data'
TIME_ZONE     = 'America/New_York'
SCHED_TIME    = '04:00'
LOG_FILE_NAME = '/var/log/data_collector.log'
VERBOSE       = 1

PID_FILE      = '/var/run/data_collector.pid'

DEV_LIST = [ 'babak.emami@gmail.com' ]

DEBUG_MODE = True

if DEBUG_MODE:
    SCHED_FLAG = False
else:
    SCHED_FLAG = True

# ***********************************************************************
# Class DataCollector: Daemon to collect and update intraday data
# ***********************************************************************

class DataCollector( Daemon ):

    def __init__(   self,
                    etfs        = ETFS,
                    stocks      = STOCKS,
                    futures     = FUTURES,
                    indexes     = INDEXES,
                    nDays       = NUM_DAYS,
                    datDir      = DAT_DIR,
                    source      = SOURCE,
                    timeZone    = TIME_ZONE,                    
                    schedTime   = SCHED_TIME,
                    logFileName = LOG_FILE_NAME,
                    verbose     = VERBOSE         ):

        Daemon.__init__( self, PID_FILE )

        self.etfs        = etfs
        self.stocks      = stocks
        self.futures     = futures
        self.indexes     = indexes
        self.nDays       = nDays
        self.datDir      = datDir
        self.source      = source
        self.timeZone    = timeZone
        self.schedTime   = schedTime
        self.logFileName = logFileName        
        self.verbose     = verbose

        if not os.path.exists( self.datDir ):
            os.makedirs( self.datDir )            
            
        self.logger = utl.getLogger( logFileName, verbose )

        devAlertHd = utl.getAlertHandler( alertLevel = logging.ERROR,
                                          subject    = 'A message for Opti-Trade developers!',
                                          mailList   = DEV_LIST )
        
        self.logger.addHandler( devAlertHd )
        
        if self.timeZone != 'America/New_York':
            self.logger.warning( 'Only America/New_York time zone is supported at this time!' )
            self.logger.warning( 'Switching to America/New_York time zone!' )
            self.timeZone = 'America/New_York'

        os.environ[ 'TZ' ] = self.timeZone
        
        self.logger.info( 'Daemon is initialized ...' )            
            
    def updateData( self ):

        typeHash = {}

        for symbol in self.etfs:
            typeHash[ symbol ] = 'ETFs'

        for symbol in self.stocks:
            typeHash[ symbol ] = 'stocks'
            
        for symbol in self.futures:
            typeHash[ symbol ] = 'futures'

        for symbol in self.indexes:
            typeHash[ symbol ] = 'indexes'        

        symbols = self.etfs +\
                  self.stocks +\
                  self.futures +\
                  self.indexes

        self.logger.info( 'Getting data for %d symbols...',
                          len( symbols ) )
        
        for symbol in symbols:

            filePath = os.path.join( self.datDir, symbol + '.pkl' )
                
            oldDf = None
            if os.path.exists( filePath ):
                oldDf = pd.read_pickle( filePath )
            
            if self.source == 'kibot':
                if typeHash[ symbol ] == 'ETFs':
                    newDf = utl.getKibotData( etfs    = [ symbol ],
                                              nDays   = self.nDays,
                                              logger  = self.logger )
                elif typeHash[ symbol ] == 'stocks':
                    newDf = utl.getKibotData( stocks  = [ symbol ],
                                              nDays   = self.nDays,
                                              logger  = self.logger )
                elif typeHash[ symbol ] == 'futures':
                    newDf = utl.getKibotData( futures = [ symbol ],
                                              nDays   = self.nDays,
                                              logger  = self.logger )
                elif typeHash[ symbol ] == 'indexes':                    
                    newDf = utl.getKibotData( indexes = [ symbol ],
                                              nDays   = self.nDays,
                                              logger  = self.logger )
                else:
                    self.logger.error( 'Unknow type %s for symbol %s',
                                       typeHash[ symbol ],
                                           symbol )
            elif self.source == 'yahoo':
                if typeHash[ symbol ] == 'ETFs':
                    newDf = utl.getYahooData( etfs    = [ symbol ],
                                              nDays   = self.nDays,
                                              logger  = self.logger )
                elif typeHash[ symbol ] == 'stocks':
                    newDf = utl.getYahooData( stocks  = [ symbol ],
                                              nDays   = self.nDays,
                                              logger  = self.logger )
                elif typeHash[ symbol ] == 'futures':
                    newDf = utl.getYahooData( futures = [ symbol ],
                                              nDays   = self.nDays,
                                              logger  = self.logger )
                elif typeHash[ symbol ] == 'indexes':                    
                    newDf = utl.getYahooData( indexes = [ symbol ],
                                              nDays   = self.nDays,
                                              logger  = self.logger )
                else:
                    self.logger.error( 'Unknow type %s for symbol %s',
                                       typeHash[ symbol ],
                                       symbol )
            else:
                self.logger.error( 'Unkown data source %s!', self.source )

            if oldDf is not None:
                newDf = newDf[ newDf.Date > oldDf.Date.max() ]
                newDf = pd.concat( [ oldDf, newDf ] )

            newDf = newDf.sort_values( 'Date' )

            self.logger.info( 'Saving data to %s...', filePath )
            
            newDf.to_pickle( filePath )

        self.logger.info( 'Done with getting data for %d symbols...',
                          len( symbols ) )
            
        return True
 
    def run( self ):

        os.environ[ 'TZ' ] = self.timeZone

        if not SCHED_FLAG:
            self.updateData()
        else:
            schedule.every().day.at( self.schedTime ).do( self.updateData )
            
            while True: 
                schedule.run_pending() 
                time.sleep( 60 )
              
# ***********************************************************************
# Run daemon
# ***********************************************************************

if __name__ ==  '__main__':

    daemon = DataCollector()

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


