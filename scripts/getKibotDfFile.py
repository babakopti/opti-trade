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

dfFile    = 'data/dfFile_kibot_all.pkl'

nDays     = 3000

futures   = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

invHash   = { 'TQQQ' : 'SQQQ',
              'SPY'  : 'SH',
              'DDM'  : 'DXD',
              'MVV'  : 'MZZ',
              'UWM'  : 'TWM',
              'SAA'  : 'SDD',
              'UYM'  : 'SMN',
              'UGE'  : 'SZK',
              'UCC'  : 'SCC',
              'FINU' : 'FINZ',
              'RXL'  : 'RXD',
              'UXI'  : 'SIJ',
              'URE'  : 'SRS',
              'ROM'  : 'REW',
              'UJB'  : 'SJB',
              'AGQ'  : 'ZSL',     
              'DIG'  : 'DUG',
              'USD'  : 'SSG',
              'ERX'  : 'ERY',
              'UYG'  : 'SKF',
              'UCO'  : 'SCO',
              'BOIL' : 'KOLD',
              'UPW'  : 'SDP',
              'UGL'  : 'GLL',
              'BIB'  : 'BIS',
              'UST'  : 'PST',
              'UBT'  : 'TBT' }

allETFs     = list( invHash.keys() ) + list( invHash.values() )

# ***********************************************************************
# Get data and save to pickle file
# ***********************************************************************

df = utl.getKibotData( etfs    = allETFs,
                       futures = futures,
                       indexes = [],
                       nDays   = nDays       )

# for item in df.columns:
#     if item == 'Date':
#         continue
#     plt.plot( df[ item ] )
#     plt.ylabel( item )
#     plt.show()

#df = df[ df.Date >= minDate ]
#df = df[ df.Date <= maxDate ]
df.to_pickle( dfFile, protocol = 4 )
