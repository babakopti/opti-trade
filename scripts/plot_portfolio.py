# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import datetime
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ***********************************************************************
# Input
# ***********************************************************************

prtFile     = 'portfolios/sign_trick_20191022.txt'
dfFile      = 'data/dfFile_2017plus.pkl'
base        = 'SPY'

prtFile     = 'portfolios/crypto_20191027.txt'
dfFile      = 'data/dfFile_crypto.pkl'
base        = 'BTC'

initAssetVal = 1000000.0
minDate      = '2018-01-03'
maxDate      = '2019-06-30'

# ***********************************************************************
# Read portfolio dates, assets
# ***********************************************************************

prtHash = ast.literal_eval( open( prtFile, 'r' ).read() )
dates   = sorted( list( prtHash.keys() ) )
assets  = list( prtHash[ dates[0] ].keys() )

# ***********************************************************************
# Get actual open prices
# ***********************************************************************

dataDf    = pd.read_pickle( dfFile )
dataDf    = dataDf.sort_values( 'Date' )
dataDf    = dataDf[ [ 'Date' ] + assets ]
dataDf    = dataDf.drop_duplicates()
dataDf    = dataDf.dropna()
dataDf    = dataDf.reset_index( drop = True )
priceHash = {}
actDates  = []

for date in dates:

    if date < minDate:
        continue

    if date > maxDate:
        continue

    begDate = pd.to_datetime( date )
    endDate = begDate + datetime.timedelta( days = 1 )
    tmpDf   = dataDf[ dataDf.Date >= begDate ]
    tmpDf   = tmpDf[ tmpDf.Date < endDate ]
    tmpDf   = tmpDf.sort_values( 'Date' )
    tmpHash = {}
    
    if tmpDf.shape[0] == 0:
        print( 'Skipping date', date )
        continue

    actDates.append( date )

    for asset in assets:
        tmpHash[ asset ] = list( tmpDf[ asset ] )[0]

    priceHash[ date ] = tmpHash.copy()

# ***********************************************************************
# Loop through portfolio and get vals
# ***********************************************************************

val     = initAssetVal
spy     = initAssetVal
vals    = [ val ]
spys    = [ val ]
nDates  = len( actDates )

for i in range( 1, nDates ):

    currDate = actDates[i]
    prevDate = actDates[i-1] 
    prevVal  = val

    for asset in assets:
        currPrice = priceHash[ currDate ][ asset ]
        prevPrice = priceHash[ prevDate ][ asset ]
        prevWt    = prtHash[ prevDate ][ asset ]
        prevQty   = round( prevWt * prevVal / prevPrice ) 
        #print( prevDate, asset, prevPrice, prevWt, prevQty )
        val      += prevQty * ( currPrice - prevPrice )

    #sys.exit()
    vals.append( val )

    currPrice = priceHash[ currDate ][ base ]
    prevPrice = priceHash[ prevDate ][ base ]
    prevWt    = 1.0
    prevQty   = round( prevWt * prevVal / prevPrice ) 
    spy      += prevQty * ( currPrice - prevPrice )

    spys.append( spy )

# ***********************************************************************
# Plot
# ***********************************************************************

x = []

for date in actDates:
    x.append( pd.to_datetime( date ) )

plt.plot( x, vals, 'b-', x, spys, 'r-' )

plt.xlabel( 'Date' )
plt.ylabel( 'Portfolio Value ($)' )
plt.legend( [ 'Algorithm', base ] )
plt.show()
