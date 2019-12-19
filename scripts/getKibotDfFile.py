# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append( os.path.abspath( '../' ) )

import utl.utils as utl

from mod.mfdMod import MfdMod

# ***********************************************************************
# Set some parameters and read data
# ***********************************************************************

dfFile    = 'data/dfFile_kibot.pkl'

minDate  = pd.to_datetime( '2015-01-01 00:00:00' )
maxDate  = pd.to_datetime( '2019-12-17 23:59:00' )

indexes     = [ 'INDU', 'NDX', 'SPX', 'COMPQ', 'RUT',  'OEX',  
                'MID',  'SOX', 'RUI', 'RUA',   'TRAN', 'HGX',  
                'TYX',  'XAU'                      ] 

ETFs        = [ 'TQQQ', 'SPY', 'DDM', 'MVV', 'UWM', 'DIG', 'USD',
                'ERX',  'UYG', 'UPW', 'UGL', 'BIB', 'UST', 'UBT'  ]
#ETFs        = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH', 
#                'SMH', 'XLE', 'XLF', 'XLU', 'EWJ'          ]

futures     = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

nDays       = ( maxDate - minDate ).days

# ***********************************************************************
# Get data and save to pickle file
# ***********************************************************************

df = utl.getKibotData( etfs    = ETFs,
                       futures = futures,
                       indexes = indexes,                                                                                
                       nDays   = nDays       )

for item in df.columns:
    if item == 'Date':
        continue
    plt.plot( df[ item ] )
    plt.ylabel( item )
    plt.show()

#df = df[ df.Date >= minDate ]
#df = df[ df.Date <= maxDate ]
df.to_pickle( dfFile, protocol = 4 )
