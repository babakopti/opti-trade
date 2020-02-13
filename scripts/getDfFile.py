# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import datetime
import numpy as np
import pandas as pd

from utils import getDf

sys.path.append( os.path.abspath( '../' ) )

from mod.mfdMod import MfdMod

from utl.utils import getKibotData

# ***********************************************************************
# Set some parameters and read data
# ***********************************************************************

quandlDir = '/Users/babak/workarea/data/quandl_data'
piDir     = '/Users/babak/workarea/data/pitrading_data'
#dfFile    = 'data/dfFile_long_term_pitrading.pkl'

dfFile    = 'data/dfFile_long_term_kibot.pkl'

#minDate  = pd.to_datetime( '2001-01-01 00:00:00' )
#maxDate  = pd.to_datetime( '2016-12-31 23:59:00' )

indices     = [ 'INDU', 'NDX', 'SPX', 'RUT', 'OEX',  
                'MID',  'SOX', 'RUI', 'RUA', 'TRAN',
                'HGX',  'TYX',  'HUI', 'XAU'               ] 

futures     = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

ETFs        = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH', 
                'SMH', 'XLE', 'XLF', 'XLU', 'EWJ'          ]

allETFs     = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'GDX', 
                'OIH', 'RSX', 'SMH', 'XLE', 'XLF', 'XLV', 
                'XLU', 'FXI', 'TLT', 'EEM', 'EWJ', 'IYR', 
                'SDS', 'SLV', 'GLD', 'USO', 'UNG', 'TNA', 
                'TZA', 'FAS'                               ]

stocks      = [ 'MMM',  'AXP', 'AAPL', 'BA', 'CAT',  'CVX',
                'CSCO', 'KO',  'XOM',  'GS',  'HD',  'INTC',
                'IBM', 'JNJ',  'JPM',  'MCD', 'MRK', 'MSFT', 
                'NKE', 'PFE',  'PG',   'TRV', 'UTX', 'UNH', 
                'VZ',  'WMT',  'WBA', 'DIS'                ]

forex       = [ 'USDJPY', 'USDCHF', 'USDCAD', 'NZDUSD',
                'GBPUSD', 'EURUSD', 'AUDUSD'               ]

velNames    = ETFs + indices + futures

# ***********************************************************************
# Get data and save to pickle file
# ***********************************************************************

#df = getDf( quandlDir, piDir, velNames )

df = getKibotData( etfs    = ETFs,
                   futures = futures,
                   indexes = indices,
                   nDays   = 3000       )

#df = df[ df.Date >= minDate ]
#df = df[ df.Date <= maxDate ]

df.to_pickle( dfFile, protocol = 4 )
