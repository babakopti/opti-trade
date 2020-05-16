# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import logging
import time
import datetime
import schedule
import pandas as pd

from google.cloud import storage

from daemonBase import Daemon

sys.path.append( os.path.abspath( '../' ) )

import utl.utils as utl

from dat.assets import ETF_HASH, SUB_ETF_HASH, NEW_ETF_HASH, POP_ETF_HASH
from dat.assets import OPTION_ETFS, PI_ETFS
from dat.assets import INDEXES, PI_INDEXES
from dat.assets import FUTURES

GOOGLE_STORAGE_JSON = '/home/babak/opti-trade/daemons/keyfiles/google_storage.json'
GOOGLE_BUCKET = 'prt-storage'
GOOGLE_PREFIX = 'data-backtup'

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

NUM_DAYS      = 5
SOURCE        = 'yahoo'
DAT_DIR       = '/var/data'
TIME_ZONE     = 'America/New_York'
SCHED_TIME    = '01:00'
LOG_FILE_NAME = '/var/log/data_collector.log'
VERBOSE       = 1

CHECK_SPLIT_MONTHS = 4

PID_FILE      = '/var/run/data_collector.pid'

DEV_LIST = [ 'babak.emami@gmail.com' ]

DEBUG_MODE = False

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
                                          subject    = 'A message from data collector!',
                                          mailList   = DEV_LIST )
        
        self.logger.addHandler( devAlertHd )
        
        if self.timeZone != 'America/New_York':
            self.logger.warning( 'Only America/New_York time zone is supported at this time!' )
            self.logger.warning( 'Switching to America/New_York time zone!' )
            self.timeZone = 'America/New_York'

        os.environ[ 'TZ' ] = self.timeZone
        
        self.logger.info( 'Daemon is initialized ...' )            

    def backupData( self, filePath ):

        self.logger.info( 'Backuping up %s on Google cloud...', filePath )            
        
        client   = storage.Client.from_service_account_json( GOOGLE_STORAGE_JSON )
        bucket   = client.get_bucket( GOOGLE_BUCKET )
        baseName = os.path.basename( filePath )
        tmpName  = GOOGLE_PREFIX + '/' + baseName
        blob     = bucket.blob( tmpName )
            
        with open( filePath, 'rb' ) as fHd:
            blob.upload_from_file( fHd )

        self.logger.info( '%s was saved to bucket!', tmpName )

    def reportSplit( self, df, symbol, months = CHECK_SPLIT_MONTHS ):

        begDate = pd.to_datetime( df.Date.max() ) - \
            pd.timedelta( months = months )
        
        splitDf = df[ ( df.Date >= begDate ) &
                      ( df[ symbol ].pct_change() <= -0.5 ) ]

        if splitDf.shape[0] > 0:

            for date in list( splitDf.Date ):
                tmpDf = df[ df.Date == date ]
                ratio = 1.0 / tmpDf[ symbol ].pct_change()
                self.logger.critical( 'Possible split of 1:%d detected for %s: \n %s',
                                      ratio,
                                      symbol,
                                      str( tmpDf ) )

        splitDf = df[ ( df.Date >= begDate ) &
                      ( df[ symbol ].pct_change() >= 1.0 ) ]                

        if splitDf.shape[0] > 0:
            
            for date in list( splitDf.Date ):
                tmpDf = df[ df.Date == date ]
                ratio = tmpDf[ symbol ].pct_change()                
                self.logger.critical( 'Possible reverse split of %d:1 detected for %s: \n %s',
                                      ratio,
                                      symbol,
                                      str( tmpDf ) )
            
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
                oldDf[ 'Date' ] = oldDf.Date.apply( pd.to_datetime )
            
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

            if newDf is None or newDf.shape[0] == 0:
                self.logger.error( 'No new data found for symbol %s!',
                                   symbol )
                continue
                
            if oldDf is not None:
                maxDt = oldDf.Date.max()
                newDf = newDf[ newDf.Date > maxDt ]
                newDf = pd.concat( [ oldDf, newDf ] )

            newDf = newDf.sort_values( 'Date' )

            self.logger.info( 'Saving data to %s...', filePath )
            
            newDf.to_pickle( filePath, protocol = 4 )

            self.reportSplit( newDf, symbol )
            
            self.backupData( filePath )

        self.logger.critical( 'Done with getting data for %d symbols...',
                              len( symbols ) )
            
        return True

    def process( self ):

        try:
            self.updateData()
        except Exception as e:
            self.logger.error( e )
            
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


