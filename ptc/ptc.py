# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import joblib
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
                    nPTAvgDays  = None,                    
                    testRatio   = 0.2,
                    method      = 'bayes',
                    minVix      = None,
                    maxVix      = None,
                    minProb     = None,                    
                    logFileName = None,                    
                    verbose     = 1          ):

        self.symbol      = symbol
        self.dfFile      = dfFile
        self.ptThreshold = ptThreshold
        self.nAvgDays    = nAvgDays
        self.nPTAvgDays  = nPTAvgDays                    
        self.testRatio   = testRatio
        self.method      = method
        self.minVix      = minVix
        self.maxVix      = maxVix
        self.minProb     = minProb
        self.logFileName = logFileName
        self.verbose     = verbose
        self.logger      = getLogger( logFileName, verbose, 'mod' )
        self.classes     = [ 'not_peak_or_trough', 'peak', 'trough' ] 
        self.dayDf       = None
        self.classifier  = None
        self.trnAccuracy = None
        self.oosAccuracy = None        
        self.normTrnMat  = None
        self.normOosMat  = None        
        
        assert method == 'bayes', 'Only Bayes method is supported right now!'
        
        self.setDf()

    def setDf( self ):

        symbol     = self.symbol
        dfFile     = self.dfFile
        thres      = self.ptThreshold
        nAvgDays   = self.nAvgDays
        nPTAvgDays = self.nPTAvgDays
        fileExt    = dfFile.split( '.' )[-1]
        
        if fileExt == 'csv':
            df = pd.read_csv( dfFile ) 
        elif fileExt == 'pkl':
            df = pd.read_pickle( dfFile ) 
        else:
            assert False, 'Unknown input file extension %s' % fileExt

        assert symbol in df.columns, 'Symbol %s not found in %s' \
            % ( symbol, dfFile )

        if self.minVix is not None or self.maxVix is not None:
                assert 'VIX' in df.columns, 'VIX not found in %s' \
                    % dfFile

        if self.minVix is not None:
            df = df[ df.VIX >= self.minVix ]
            
        if self.maxVix is not None:
            df = df[ df.VIX <= self.maxVix ]

        df[ 'Date' ]  = df.Date.apply( pd.to_datetime )
        df[ 'Date0' ] = df.Date.apply( lambda x : x.strftime( '%Y-%m-%d' ) )
        
        dayDf = df.groupby( 'Date0', as_index = False )[ symbol ].mean()
        dayDf = dayDf.rename( columns = { 'Date0' : 'Date' } )

        dayDf[ 'vel' ] = np.gradient( dayDf[ symbol ], 2 )
        dayDf[ 'acl' ] = np.gradient( dayDf[ 'vel' ], 2 )

        if nPTAvgDays is not None:
            dayDf[ 'avgBkd' ] = dayDf[ symbol ].\
                rolling( min_periods = 1,
                         window = nPTAvgDays ).mean()
            dayDf[ 'avgFwd' ] = dayDf[ symbol ].shift( -nPTAvgDays ).\
                rolling( min_periods = 1,
                         window = nAvgDays ).mean()        
        dayDf[ 'avgAcl' ] = dayDf.acl.rolling( min_periods = 1,
                                                window = nAvgDays ).mean()

        dayDf[ 'feature' ] = dayDf.acl - dayDf.avgAcl

        if nPTAvgDays is None:
            tmpDf = dayDf[ ( ( dayDf[ symbol ] - dayDf[ symbol ].shift(1) ) >
                             thres * dayDf[ symbol ] ) &
                           ( ( dayDf[ symbol ] - dayDf[ symbol ].shift(-1) ) >
                             thres * dayDf[ symbol ] ) ][ [ 'Date' ] ]
        else:
            tmpDf = dayDf[ ( ( dayDf[ symbol ] - dayDf.avgBkd ) >
                             thres * dayDf[ symbol ] ) &
                           ( ( dayDf[ symbol ] - dayDf.avgFwd ) >
                             thres * dayDf[ symbol ] ) ][ [ 'Date' ] ]        

        tmpDf[ 'peak' ] = PEAK
        
        dayDf = dayDf.merge( tmpDf, on = 'Date', how = 'left' )

        if nPTAvgDays is None:        
            tmpDf = dayDf[ ( ( dayDf[ symbol ] - dayDf[ symbol ].shift(1) ) <
                             -thres * dayDf[ symbol ] ) &
                           ( ( dayDf[ symbol ] - dayDf[ symbol ].shift(-1) ) <
                             -thres * dayDf[ symbol ] ) ][ [ 'Date' ] ]
        else:
            tmpDf = dayDf[ ( ( dayDf[ symbol ] - dayDf.avgBkd ) <
                             thres * dayDf[ symbol ] ) &
                           ( ( dayDf[ symbol ] - dayDf.avgFwd ) <
                             thres * dayDf[ symbol ] ) ][ [ 'Date' ] ]        
        
        tmpDf[ 'trough' ] = TROUGH

        dayDf = dayDf.merge( tmpDf, on = 'Date', how = 'left' )

        dayDf[ 'peak' ]   = dayDf.peak.fillna( NOT_PT )
        dayDf[ 'trough' ] = dayDf.trough.fillna( NOT_PT )        
        dayDf[ 'ptTag' ]  = dayDf[ 'peak' ] + dayDf[ 'trough' ]
        dayDf[ 'ptTag' ]  = dayDf.ptTag.apply( lambda x : int( x ) )

        dayDf = dayDf[ dayDf.ptTag.isin( [PEAK, TROUGH, NOT_PT] ) ]

        assert set( dayDf.ptTag ) == { PEAK, TROUGH, NOT_PT }, \
            'Unexpected peak/trough tag!'

        self.logger.info( 'A total of %d samples for %s! '
                          'Found %d peaks and %d troughs!',
                          dayDf.shape[0],
                          symbol,
                          dayDf[ dayDf.ptTag == PEAK ].shape[0],
                          dayDf[ dayDf.ptTag == TROUGH ].shape[0]  )

        dayDf[ 'Date' ] = dayDf.Date.astype( 'datetime64[ns]' )
        
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

    def plotSymbol( self,
                    actPeaks   = False,
                    actTroughs = False,
                    prdPeaks   = False,
                    prdTroughs = False   ):                    

        symbol  = self.symbol        
        df      = self.dayDf
        legends = [ 'All' ]
        
        plt.plot( df[ 'Date' ], df[ symbol ], 'b-' )

        if actPeaks:
            plt.plot( df[ df.ptTag == PEAK ][ 'Date' ],
                      df[ df.ptTag == PEAK ][ symbol ],
                      'go' )
            legends.append( 'Peaks' )

        if actTroughs:
            plt.plot( df[ df.ptTag == TROUGH ][ 'Date' ],
                      df[ df.ptTag == TROUGH ][ symbol ],
                      'ys' )
            legends.append( 'Troughs' )

        if prdPeaks:
            plt.plot( df[ df.ptTagPrd == PEAK ][ 'Date' ],
                      df[ df.ptTagPrd == PEAK ][ symbol ],
                      'r.' )
            legends.append( 'Predicted Peaks' )

        if prdTroughs:
            plt.plot( df[ df.ptTagPrd == TROUGH ][ 'Date' ],
                      df[ df.ptTagPrd == TROUGH ][ symbol ],
                      'c.' )
            legends.append( 'Predicted Troughs' )
            
        plt.legend( legends )
        plt.ylabel( symbol )
        plt.show()
        
    def classify( self ):

        self.logger.info( 'Building a peak/trough classifier for %s',
                          self.symbol )
        
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
        
        self.classifier  = gnb
        self.trnAccuracy = gnb.score( XTrn, yTrn )
        self.oosAccuracy = gnb.score( XOos, yOos )
        
        self.logger.info( 'In-sample accuracy: %0.4f',
                          self.trnAccuracy )
        self.logger.info( 'Out-of-sample accuracy: %0.4f',
                          self.oosAccuracy )

        yTrnPred = gnb.predict( XTrn )        
        confMat  = confusion_matrix( yTrn, yTrnPred )

        self.logger.info( 'In-sample confusion matrix:\n %s',
                          str( confMat ) )

        self.normTrnMat = self.normalizeConfMat( confMat )     
                
        self.logger.info( 'In-sample norm. confusion matrix:\n %s',
                          str( self.normTrnMat ) )

        yOosPred = gnb.predict( XOos )        
        confMat  = confusion_matrix( yOos, yOosPred )

        self.logger.info( 'Out-of-sample confusion matrix:\n %s',
                          str( confMat ) )

        self.normOosMat = self.normalizeConfMat( confMat )
        
        self.logger.info( 'Out-of-sample norm. confusion matrix:\n %s',
                          str( self.normOosMat ) )
        
        self.setPrd()                        

    def normalizeConfMat( self, confMat ):

        normMat = np.zeros( shape = ( 3, 3 ), dtype = 'd' )
        
        for i in range( 3 ):
            tmp = sum( confMat[i] )

            if tmp > 0:
                tmp = 1.0 / tmp

            for j in range( 3 ):
                normMat[i][j] = confMat[i][j] * tmp

        return normMat
        
    def setPrd( self ):

        self.dayDf[ 'ptTagPrd' ] = \
            self.dayDf.feature.apply( lambda x : self.getClass( x ) )
        self.dayDf[ 'prdProb' ] = \
            self.dayDf.feature.apply( lambda x : self.getClassProb( x ) )
        
    def getClass( self, val ):

        X = np.array( [ val ] ).reshape( ( 1, 1 ) )
        
        return self.classifier.predict( X )[0]

    def getClassProb( self, val ):

        X     = np.array( [ val ] ).reshape( ( 1, 1 ) )
        probs = self.classifier.predict_proba( X )[0]
        
        return max( probs )

    def getTrnMerits( self ):
        
        return self.trnAccuracy, self.normTrnMat

    def getOosMerits( self ):
        
        return self.oosAccuracy, self.normOosMat
    
    def save( self, fileName ):

        self.logger.info( 'Saving the peak / trough classifier to %s...',
                          fileName )
        
        with open( fileName, 'wb' ) as fHd:
            pk.dump( self.classifier, fHd, pk.HIGHEST_PROTOCOL ) 
