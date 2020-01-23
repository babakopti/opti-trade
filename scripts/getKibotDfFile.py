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
from dat.assets import ETF_HASH

# ***********************************************************************
# Set some parameters and read data
# ***********************************************************************

dfFile    = 'data/dfFile_kibot_all_popular.pkl'

nDays     = 3000

futures   = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

allETFs     = list( ETF_HASH.keys() ) 

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
