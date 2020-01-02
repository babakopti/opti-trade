# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import datetime
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append( '../' )

import utl.utils as utl

# ***********************************************************************
# Input
# ***********************************************************************

prtFile     = 'portfolios/portfolio_crash_macd.txt'
dfFile      = 'data/dfFile_2005_2010.pkl'
base        = 'SPY'

initTotVal  = 1000000.0

# ***********************************************************************
# Read portfolio dates, assets
# ***********************************************************************

prtWtsHash = ast.literal_eval( open( prtFile, 'r' ).read() )

# ***********************************************************************
# Get actual open prices
# ***********************************************************************

retDf = utl.calcPrtReturns( prtWtsHash, dfFile, initTotVal )

# ***********************************************************************
# Plot
# ***********************************************************************

plt.plot( retDf.Date, retDf.Value )
plt.xlabel( 'Date' )
plt.ylabel( 'Value' )
plt.title( prtFile )
plt.show()
