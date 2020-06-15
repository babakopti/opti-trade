# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import datetime
import logging
import numpy as np
import pandas as pd

sys.path.append( os.path.abspath( '../' ) )

import utl.utils as utl

from dat.assets import OPTION_ETFS as ETFS
from dat.assets import FUTURES, INDEXES

# ***********************************************************************
# Set some parameters 
# ***********************************************************************

dfFile       = 'data/optionTestDfFile.pkl'
minDate      = pd.to_datetime( '2012-01-01' )
maxDate      = pd.to_Datetime( '2019-01-05' )
INDEXES      = INDEXES + [ 'VIX' ]

# ***********************************************************************
# Build the dfFile
# ***********************************************************************

piDf = utl.mergePiSymbols( symbols = INDEXES + ETFS + FUTURES,
                           datDir  = '/var/pi_data',
                           minDate = minDate )
df   = utl.mergeSymbols( symbols = symbols,
                         datDir  = '/var/data',
                         fileExt = 'pkl',
                         minDate = minDate,
                         maxDate = maxDate  )
 
piDf = piDf[ piDf.Date < df.Date.min() ]
df   = pd.concat( [ piDf, df ] )        
        
df.to_pickle( dfFile )
