# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import datetime
import pytz
import dill
import logging
import pickle
import numpy as np
import pandas as pd

from collections import defaultdict

sys.path.append( os.path.abspath( '../' ) )

import utl.utils as utl

from dat.assets import OPTION_ETFS as ETFS
from dat.assets import FUTURES, INDEXES
from mod.mfdMod import MfdMod
from prt.prt import MfdOptionsPrt
from brk.tdam import Tdam

# ***********************************************************************
# Set some parameters 
# ***********************************************************************

optionDfFile = 'data/relevant_option_samples.pkl'
maxDays = 6 * 30

INDEXES  = INDEXES + [ 'VIX' ]
ASSETS   = ETFS

# ***********************************************************************
# Read and filter options chains
# ***********************************************************************

df = pd.read_pickle( optionDfFile )

df['DataDate']   = df.DataDate.apply( lambda x: pd.to_datetime(x) )
df['Expiration'] = df.Expiration.apply( lambda x: pd.to_datetime(x) )

df[ 'horizon' ] = df['Expiration'] - df['DataDate']
df[ 'horizon' ] = df.horizon.apply( lambda x : x.days )

df = df[ df.horizon <= maxDays ]

df = df[ df.Last > 0 ]

dates = list( set( df.DataDate ) )
