# ***********************************************************************                                                                   
# Import libraries                                                                                                                          
# ***********************************************************************

import sys
import os
import dill
import datetime
import time
import datetime
import pytz
import numpy as np
import pandas as pd

sys.path.append( os.path.abspath( '../' ) )

import utl.utils as utl

from mod.mfdMod import MfdMod
from prt.prt import MfdOptionsPrt
from brk.tdam import Tdam

# ***********************************************************************                                                                   
# Input parameters
# ***********************************************************************

dataFlag    = False
modFlag     = False
baseDfFile  = 'data/dfFile_long_term_all.pkl'
timeZone    = 'America/New_York'
nOosMinutes = 60
nTrnYears   = 10
factor      = 1.0e-5
refToken    = None

# ***********************************************************************                                                                   
# Input investment parameters
# ***********************************************************************

cash      = 1000
maxPriceC = 0.50 * cash
maxPriceA = 0.50 * cash
oMonths   = 12
minProb   = 0.5

# ***********************************************************************                                                                   
# Input variables
# ***********************************************************************

indexes = [ 'NDX', 'SPX', 'RUT', 'OEX',
            'MID',  'SOX', 'RUI', 'RUA', 'TRAN',
            'HGX',  'TYX', 'XAU' ]

futures = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

ETFs    = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH',
            'SMH', 'XLE', 'XLF', 'XLU', 'EWJ' ]

# ***********************************************************************                                                                   
# Set time zone
# ***********************************************************************

os.environ[ 'TZ' ] = timeZone

# ***********************************************************************                                                                   
# Set model build date
# ***********************************************************************

if dataFlag or modFlag:
    modDate = datetime.datetime.now()
    modDate = modDate.strftime( '%Y-%m-%d %H:%M:%S' )
else:
    modDate = '2020-03-12 23:44:51'

modDate = pd.to_datetime( modDate )

# ***********************************************************************                                                                   
# Set some parameters
# ***********************************************************************

dateStr    = modDate.strftime( '%Y-%m-%d_%H:%M:%S' )
dfFile     = 'data/dfFile_long_term_' + dateStr + '.pkl'
modFile    = 'models/model_long_term_' + dateStr + '.dill'
outFile    = 'chosen_options_' + dateStr + '.csv'
velNames   = ETFs + indexes + futures
minTrnDate = modDate - pd.DateOffset( years = nTrnYears )

if refToken is None:
    with open( 'ref_token.txt', 'r' ) as fHd:
        refToken = tmp = fHd.read()[:-1]

# ***********************************************************************                                                                   
# Get data
# ***********************************************************************

if dataFlag:
    cols     = [ 'Date' ] + velNames

    oldDf    = pd.read_pickle( baseDfFile )
    newDf    = utl.getKibotData( etfs    = ETFs,
                                 futures = futures,
                                 indexes = indexes,
                                 nDays   = 30        )

    oldDf    = oldDf[ cols ]
    newDf    = newDf[ cols ]
    newDf    = newDf[ newDf.Date > oldDf.Date.max() ]
    allDf    = pd.concat( [ oldDf, newDf ] )
    allDf    = allDf[ allDf.Date >= minTrnDate ]

    allDf.to_pickle( dfFile, protocol = 4 )

# ***********************************************************************                                                                   
# Build the model
# ***********************************************************************

if modFlag:
    maxOosDate = pd.read_pickle( dfFile ).Date.max()

    if maxOosDate != modDate:
        print( 'Warning maxOosDate is not the same as modDate: %s vs. %s' \
               % ( maxOosDate, modDate ) )

    maxTrnDate = maxOosDate - datetime.timedelta( minutes = nOosMinutes )
    
    mfdMod     = MfdMod( dfFile       = dfFile,
                         minTrnDate   = minTrnDate,
                         maxTrnDate   = maxTrnDate,
                         maxOosDate   = maxOosDate,
                         velNames     = velNames,
                         maxOptItrs   = 500,
                         optGTol      = 1.0e-2,
                         optFTol      = 1.0e-2,
                         factor       = factor,
                         regCoef      = 1.0e-3,
                         smoothCount  = None,
                         logFileName  = None,
                         verbose      = 1        )

    validFlag = mfdMod.build()

    print( 'Success :', validFlag )

    mfdMod.ecoMfd.lighten()

    mfdMod.save( modFile )

# ***********************************************************************                                                                   
# Get TD Ameritrade handle
# ***********************************************************************

td = Tdam( refToken = refToken )

# ***********************************************************************                                                                   
# Get options chain
# ***********************************************************************

options = []

for symbol in ETFs:
    
    print( 'Getting options for %s...' % symbol )
    
    tmpList = td.getOptionsChain( symbol )
    options += tmpList
    
print( 'Found %d options contracts!' % len( options ) )

# ***********************************************************************                                                                   
# Get asset prices
# ***********************************************************************

assetHash = {}

for symbol in ETFs:
    assetHash[ symbol ] = td.getQuote( symbol )

for symbol in futures:
    print( symbol )
    val, date = utl.getYahooLastValue( symbol,
                                       sType = 'futures' )
    print( val, date )
    assetHash[ symbol ] = val

for symbol in indexes:
    print( symbol )
    val, date = utl.getYahooLastValue( symbol,
                                       sType = 'index' )
    print( val, date )
    assetHash[ symbol ] = val

print( assetHash )

# ***********************************************************************                                                                   
# Instantiate options portfolio
# ***********************************************************************

snapDate = datetime.datetime.now()
snapDate = snapDate.strftime( '%Y-%m-%d %H:%M:%S' )
snapDate = pd.to_datetime( snapDate )
minDate  = snapDate + pd.DateOffset( days   = 1 )
maxDate  = snapDate + pd.DateOffset( months = oMonths )

prtObj  = MfdOptionsPrt( modFile     = modFile,
                         assetHash   = assetHash,
                         curDate     = snapDate,
                         minDate     = minDate,
                         maxDate     = maxDate,
                         minProb     = minProb,
                         rfiDaily    = 0.0,
                         tradeFee    = 0.5,
                         nDayTimes   = 1140,
                         logFileName = None,                    
                         verbose     = 1          ) 

# ***********************************************************************                                                                   
# Get action 
# ***********************************************************************

selHash = prtObj.selOptions( options   = options,
                             cash      = cash,
                             maxPriceC = maxPriceC,
                             maxPriceA = maxPriceA,
                             maxCands    = None         )


symbols = []
cnts    = []

for symbol in selHash:
    symbols.append( symbol )
    cnts.append( selHash[ symbol ] )

actDf = pd.DataFrame( { 'symbol' : symbols, 'count' : cnts } )

print( actDf )

print( actDf.shape )

actDf.to_csv( outFile, index = False )
