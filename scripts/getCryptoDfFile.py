# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import datetime
import numpy as np
import pandas as pd

sys.path.append( os.path.abspath( '../' ) )

import utl.utils as utl

from dat.assets import INDEXES

# ***********************************************************************
# Set some parameters and read data
# ***********************************************************************

minDate = pd.to_datetime( '2018-12-01 00:00:00' )

indexes    = INDEXES + [ 'VIX' ]
cryptos    = [ 'BTC', 'ETH', 'LTC', 'ZEC' ]
baseDatDir = '/var/data'
dfFile     = 'data/dfFile_crypto.pkl'

# ***********************************************************************
# Get data and save to pickle file
# ***********************************************************************

indexDf = utl.mergeSymbols( symbols = indexes,
                            datDir  = baseDatDir,
                            fileExt = 'pkl',
                            minDate = minDate,
                            logger  = None   )

indexDf.Date = indexDf.Date.dt.tz_localize( 'America/New_York' )
indexDf.Date = indexDf.Date.dt.tz_convert( 'UTC' )
indexDf.Date = indexDf.Date.dt.tz_convert( None )

cryptoDf = utl.mergeSymbols( symbols = cryptos,
                             datDir  = baseDatDir,
                             fileExt = 'pkl',
                             minDate = minDate,
                             logger  = None   )

outDf = indexDf.merge( cryptoDf, how = 'outer', on = 'Date' )
outDf = outDf.interpolate( method = 'linear' )
outDf = outDf.dropna()
outDf = outDf.sort_values( 'Date' )
outDf = outDf.reset_index( drop = True )

outDf.to_pickle( dfFile, protocol = 4 )
