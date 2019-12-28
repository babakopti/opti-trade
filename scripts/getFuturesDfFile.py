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

dfFile  = 'data/dfFile_futures.pkl'

minDate = pd.to_datetime( '2015-01-01 00:00:00' )
maxDate = pd.to_datetime( '2019-12-25 23:59:00' )

indexes = [ 'INDU', 'NDX', 'SPX', 'COMPQ', 'RUT',  'OEX',  
            'MID',  'SOX', 'RUI', 'RUA',   'TRAN', 'HGX',  
            'TYX'                      ] 

# fuDf = pd.read_csv( 'data/Futures_kibot.txt', delimiter = '\t' )

# fuDf[ 'Continuous' ] = fuDf.Description.apply( lambda x : 'CONTINUOUS' in x )

# fuDf = fuDf[ fuDf.Continuous == True ]

# fuDf = fuDf[ [ 'Base', 'StartDate', 'Description' ] ]

# fuDf[ 'StartDate' ] = fuDf[ 'StartDate' ].apply( pd.to_datetime )

# fuDf.reset_index( drop = True, inplace = True )

# fuDf[ fuDf.StartDate <= pd.to_datetime( '2010-01-01' ) ].shape

#futures = list( set( fuDf.Base ) - set( [ 'RTY', 'TN', 'BTC', 'SIR', 'SIL'  ] ) )

futures     = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM', 'CL', 'NG',
                'GC', 'SI', 'TY', 'FV', 'TU', 'C', 'HG', 'S', 'W', 'RB',
                'BO', 'O' ]

nDays   = ( maxDate - minDate ).days

# ***********************************************************************
# Get data and save to pickle file
# ***********************************************************************

df = utl.getKibotData( futures = futures,
                       indexes = indexes,                                                                                
                       nDays   = nDays       )

df.to_pickle( dfFile, protocol = 4 )
