# ***********************************************************************
# Import libs
# ***********************************************************************

import os, sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from utils import getDf

# ***********************************************************************
# Some inputs
# ***********************************************************************

dfFile = 'data/dfFile_2017plus.pkl'
qlDir  = '/Users/babak/workarea/data/quandl_data'
piDir  = '/Users/babak/workarea/data/pitrading_data'
vars   = [ 'XLU', 'SPX', 'TYX' ]
sVar   = 'XLU'

# ***********************************************************************
# Linear reg stuff
# ***********************************************************************

if False:

    minTrnDt  = pd.to_datetime( '2017-01-01' )
    maxTrnDt  = pd.to_datetime( '2018-12-31' )

    df = pd.read_pickle( dfFile )

    df[ 'Date' ] = df.Date.apply( lambda x : pd.to_datetime(x).date() )

    df = df.groupby( 'Date', as_index = False )[ vars ].mean() 
    
    df[ 'Date' ] = df.Date.apply( lambda x : pd.to_datetime(x) )

    df = df.reset_index( drop = True )

    trnDf = df[ df.Date >= minTrnDt ][ df.Date <= maxTrnDt ]
    yTrn  = np.array( trnDf[ sVar ] )
    nTrn  = trnDf.shape[0]
    yPrd  = df[ sVar ].rolling( 120, 
                               win_type = 'blackman', 
                               center = True ).mean()
    yLow  = 0.95 * yPrd
    yHigh = 1.05 * yPrd

    matplotlib.rcParams.update({'font.size': 16})

    plt.plot( trnDf.Date, yTrn, 'bo', df.Date, yPrd, 'r-', linewidth = 2.0 )
    plt.xlabel( 'Date' )
    plt.ylabel( sVar )
    plt.legend( [ 'Actual', 'Predicted' ] )
    plt.xlim( pd.to_datetime( '2018-01-01' ),
              pd.to_datetime( '2019-04-01' ) )

    plt.fill_between( df.Date, yLow, yHigh, alpha = 0.2 ) 
    plt.show()
  
# ***********************************************************************
# Economic universe
# ***********************************************************************

if True:
    
    minTrnDt  = pd.to_datetime( '2017-01-01' )
    maxTrnDt  = pd.to_datetime( '2018-12-28' )

    maxPrdDt  = pd.to_datetime( '2019-02-28' )

    df = pd.read_pickle( dfFile )
    df = df[ df.Date <= maxPrdDt ]
    
    df[ 'Date' ] = df.Date.apply( lambda x : pd.to_datetime(x).date() )

    df = df.groupby( 'Date', as_index = False )[ vars ].mean() 
    
    df[ 'Date' ] = df.Date.apply( lambda x : pd.to_datetime(x) )

    df = df.reset_index( drop = True )

    trnDf = df[ df.Date >= minTrnDt ]
    trnDf = trnDf[ trnDf.Date <= maxTrnDt ]
    yTrn1 = np.array( trnDf[ vars[0] ] )
    yTrn2 = np.array( trnDf[ vars[1] ] )
    yTrn3 = np.array( trnDf[ vars[2] ] )
    nTrn  = trnDf.shape[0]
    yPrd1 = trnDf[ vars[0] ].rolling( 10, 
                                      win_type = 'blackman', 
                                      center = True ).mean()
    yPrd2 = trnDf[ vars[1] ].rolling( 10, 
                                      win_type = 'blackman', 
                                      center = True ).mean()
    yPrd3 = trnDf[ vars[2] ].rolling( 10, 
                                      win_type = 'blackman', 
                                      center = True ).mean()
    yPrd1 = np.array( yPrd1 )
    yPrd2 = np.array( yPrd2 )
    yPrd3 = np.array( yPrd3 )

    fig = plt.figure()
    ax  = fig.add_subplot( 111, projection = '3d' )

    ax.plot( yTrn1, yTrn2, yTrn3, 'b.' )
    ax.scatter( np.array([ yPrd1[0], yPrd1[-1] ]),
                np.array([ yPrd2[0], yPrd2[-1] ]),
                np.array([ yPrd3[0], yPrd3[-1] ]), 
                marker = '^', 
                c = 'r' )
    ax.plot( yPrd1, yPrd2, yPrd3, 'r-' )

    ax.set_xlabel( vars[0] )
    ax.set_ylabel( vars[1] )
    ax.set_zlabel( vars[2] )

    if False:
        ax.axes.set_xlim3d( left   = 30,   right = 70 )
        ax.axes.set_ylim3d( bottom = 2000, top = 3300 ) 
        ax.axes.set_zlim3d( bottom = 10,   top = 50   )
    
    plt.show()



