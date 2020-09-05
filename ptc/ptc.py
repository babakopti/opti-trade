# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import dill
import pickle as pk
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
        
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

sys.path.append( os.path.abspath( '../' ) )

from utl.utils import getLogger

# ***********************************************************************
# Some definitions
# ***********************************************************************

NOT_PT = 0
PEAK   = 1
TROUGH = 2

# ***********************************************************************
# Class PTClassifier: A peak / trough classifier
# ***********************************************************************

class PTClassifier:

    def __init__(   self,
                    symbol,
                    dfFile,
                    ptThreshold = 1.0e-2,
                    nAvgDays    = 7,
                    testRatio   = 0.2,
                    method      = 'bayes',
                    logFileName = None,                    
                    verbose     = 1          ):

        self.symbol      = symbol
        self.dfFile      = dfFile
        self.ptThreshold = ptThreshold
        self.nAvgDays    = nAvgDays
        self.testRatio   = testRatio
        self.method      = method
        self.logFileName = logFileName
        self.verbose     = verbose
        self.logger      = getLogger( logFileName, verbose, 'mod' )
        self.classes     = [ 'not_peak_or_trough', 'peak', 'trough' ] 
        self.dayDf       = None
        self.classifier  = None 
        
        assert method == 'bayes', 'Only Bayes method is supported right now!'
        
        self.setDf()

    def setDf( self ):

        symbol   = self.symbol
        dfFile   = self.dfFile
        thres    = self.ptThreshold
        nAvgDays = self.nAvgDays
        fileExt  = dfFile.split( '.' )[-1]
        
        if fileExt == 'csv':
            df = pd.read_csv( dfFile ) 
        elif fileExt == 'pkl':
            df = pd.read_pickle( dfFile ) 
        else:
            assert False, 'Unknown input file extension %s' % fileExt

        assert symbol in df.columns, 'Symbol %s not found in %s' \
            % ( symbol, dfFile )

        df[ 'Date' ]  = df.Date.apply( pd.to_datetime )
        df[ 'Date0' ] = df.Date.apply( lambda x : x.strftime( '%Y-%m-%d' ) )

        dayDf = df.groupby( 'Date0', as_index = False )[ symbol ].mean()
        dayDf = dayDf.rename( columns = { 'Date0' : 'Date' } )

        dayDf[ 'vel' ] = np.gradient( dayDf[ symbol ], 2 )
        dayDf[ 'acl' ] = np.gradient( dayDf[ 'vel' ], 2 )

        dayDf[ 'avgAcl' ] = dayDf.acl.rolling( min_periods = 1,
                                                window = nAvgDays ).mean()

        # valInv = np.std( dayDf.acl )

        # if valInv > 0:
        #     valInv = 1.0 / valInv
        
        dayDf[ 'feature' ] = dayDf.acl - dayDf.avgAcl
            
        tmpDf = dayDf[ ( ( dayDf[ symbol ] - dayDf[ symbol ].shift(1) ) >
                         thres * dayDf[ symbol ] ) &
                       ( ( dayDf[ symbol ] - dayDf[ symbol ].shift(-1) ) >
                         thres * dayDf[ symbol ] ) ][ [ 'Date' ] ]
        
        tmpDf[ 'peak' ] = PEAK
        
        dayDf = dayDf.merge( tmpDf, on = 'Date', how = 'left' )

        tmpDf = dayDf[ ( ( dayDf[ symbol ] - dayDf[ symbol ].shift(1) ) <
                         -thres * dayDf[ symbol ] ) &
                       ( ( dayDf[ symbol ] - dayDf[ symbol ].shift(-1) ) <
                         -thres * dayDf[ symbol ] ) ][ [ 'Date' ] ]
        
        tmpDf[ 'trough' ] = TROUGH

        dayDf = dayDf.merge( tmpDf, on = 'Date', how = 'left' )

        dayDf[ 'peak' ]   = dayDf.peak.fillna( NOT_PT )
        dayDf[ 'trough' ] = dayDf.trough.fillna( NOT_PT )        
        dayDf[ 'ptTag' ]  = dayDf[ 'peak' ] + dayDf[ 'trough' ] 

        assert set( dayDf.ptTag ) == { PEAK, TROUGH, NOT_PT }, \
            'Unexpected peak/trough tag!'
        
        self.dayDf = dayDf
            
    def plotDists( self ):

        df = self.dayDf
        
        sns.distplot( df[ df.ptTag == PEAK ][ 'feature' ] )
        sns.distplot( df[ df.ptTag == TROUGH ][ 'feature' ] )
        sns.distplot( df[ df.ptTag == NOT_PT ][ 'feature' ] )
        
        plt.legend( [ 'Peak',
                      'Trough',
                      'Not Peak or Trough' ] )
        plt.xlabel( 'Feature' )
        plt.ylabel( 'Distribution' + self.symbol )
        plt.show()

    def plotSymbol( self ):

        symbol = self.symbol        
        df     = self.dayDf

        plt.plot( df[ symbol ], 'g-' )
        plt.plot( df[ df.ptTag == PEAK ][ symbol ], 'ro' )
        plt.plot( df[ df.ptTag == TROUGH ][ symbol ], 'bs' )
        plt.legend( [ 'All', 'Peaks', 'Troughs' ] )
        plt.ylabel( symbol )
        plt.show()
        
    def classify( self ):

        df = self.dayDf
        X  = np.array( df.feature )
        X  = X.reshape( ( len( X ), 1 ) )
        y  = np.array( df.ptTag )
        
        XTrn, XOos, yTrn, yOos = train_test_split(
            X,
            y,
            test_size    = self.testRatio,
            random_state = 0
        )

        gnb = GaussianNB()
        gnb = gnb.fit( XTrn, yTrn )

        yTrnPred = gnb.predict( XTrn )
        yOosPred = gnb.predict( XOos )

        nTrn = len( yTrn )
        
        assert nTrn > 0, 'No trainig data!'

        nSuccess  = 0
        for i in range( nTrn ):

            if yTrn[i] == yTrnPred[i]:
                nSuccess += 1
        
        self.logger.info( 'In-sample success rate; '
                          '%d out of %d!',
                          nSuccess,
                          nTrn )

        nOos = len( yOos )
        
        assert nOos > 0, 'No test data!'
        
        nSuccess = 0 
        for i in range( nOos ):

            if yOos[i] == yOosPred[i]:
                nSuccess += 1
        
        self.logger.info( 'Out-of-sample success rate; '
                          '%d out of %d!',
                          nSuccess,
                          nOos )

        yPred = gnb.predict( X )
        
        self.logger.info( '\n' + str( confusion_matrix( y, yPred ) ) )
            
        self.classifier = gnb
        
    def getClass( self, val ):

        X = np.array( [ val ] ).reshape( ( 1, 1 ) )
        
        return self.classifier.predict( X )
        
