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
# Input variables
# ***********************************************************************

indexes = [ 'INDU', 'NDX', 'SPX', 'RUT', 'OEX',
            'MID',  'SOX', 'RUI', 'RUA', 'TRAN',
            'HGX',  'TYX', 'XAU' ]

futures = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

ETFs    = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH',
            'SMH', 'XLE', 'XLF', 'XLU', 'EWJ' ]

# ***********************************************************************                                                                   
# Input parameters
# ***********************************************************************

baseDfFile  = 'data/dfFile_long_term_pitrading.pkl'
timeZone    = 'America/New_York'
minTrnDate  = '2003-01-01'
nOosMinutes = 60
oMonths     = 3
factor      = 1.0e-5

refToken = None

try:
    with open( 'ref_token.txt', 'r' ) as fHd:
        refToken = tmp = fHd.read()[:-1]
except Exception as e:
    print( e )

# ***********************************************************************                                                                   
# Set snapdate
# ***********************************************************************

os.environ[ 'TZ' ] = timeZone

snapDate = datetime.datetime.now()
snapDate = snapDate.strftime( '%Y-%m-%d %H:%M:%S' )
snapDate = pd.to_datetime( snapDate )

# ***********************************************************************                                                                   
# Get data
# ***********************************************************************

dfFile   = 'data/dfFile_long_term_' + str( snapDate ) + '.pkl'

velNames = ETFs + indexes + futures

cols     = [ 'Date' ] + velNames

oldDf    = pd.read_pickle( baseDfFile )
newDf    = utl.getKibotData( etfs    = ETFs,
                             futures = futures,
                             indexes = indexes,
                             nDays   = 3000       )

oldDf    = oldDf[ cols ]
newDf    = newDf[ cols ]
newDf    = newDf[ newDf.Date > oldDf.Date.max() ]
allDf    = pd.concat( [ oldDf, newDf ] )

allDf.to_pickle( dfFile, protocol = 4 )

# ***********************************************************************                                                                   
# Build the model
# ***********************************************************************

maxOosDate = snapDate
maxTrnDate = maxOosDate - datetime.timedelta( minutes = nOosMinutes )

modFile    = 'models/model_long_term_' + str( snapDate ) + '.dill'

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
                     verbose      = 1          )

validFlag = mfdMod.build()

print( 'Success :', validFlag )

mfd.ecoMfd.lighten()

mfdMod.save( modFileName )

# ***********************************************************************                                                                   
# Get TD Ameritrade handle
# ***********************************************************************

td = Tdam( refToken = refToken )

# ***********************************************************************                                                                   
# Get asset prices
# ***********************************************************************

assetHash = {}

for symbol in velNames:
    assetHash[ symbol ] = td.getQuote( symbol )

# ***********************************************************************                                                                   
# Instantiate options portfolio
# ***********************************************************************

maxDate = snapDate + pd.DateOffset( months = oMonths )

prtObj  = MfdOptionsPrt( modFile     = modFile,
                         assetHash   = assetHash,
                         curDate     = snapDate,
                         maxDate     = maxDate,
                         maxPriceC   = 2000.0,
                         maxPriceA   = 4000.0,
                         minProb     = 0.75,
                         rfiDaily    = 0.0,
                         tradeFee    = 0.0,
                         nDayTimes   = 1140,
                         logFileName = None,                    
                         verbose     = 1          )                        

print( 'Found %d eligible contracts..' % \
       len( prtObj.sortOptions( options ) ) )

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
# Get action 
# ***********************************************************************

actDf = prtObj.getActionDf( cash, options )

print( actDf )

actDf.to_csv( 'actDf.csv', index = False )
