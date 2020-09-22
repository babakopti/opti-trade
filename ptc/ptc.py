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
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression

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
                    symFile,
                    vixFile     = None,
                    ptThreshold = 1.0e-2,
                    nPTAvgDays  = None,                    
                    testRatio   = 0.2,
                    method      = 'bayes',
                    minVix      = None,
                    maxVix      = None,
                    minProb     = None,                    
                    logFileName = None,                    
                    verbose     = 1          ):

        self.symbol      = symbol
        self.symFile     = symFile
        self.vixFile     = vixFile
        self.ptThreshold = ptThreshold
        self.nPTAvgDays  = nPTAvgDays                    
        self.testRatio   = testRatio
        self.method      = method
        self.minVix      = minVix
        self.maxVix      = maxVix
        self.minProb     = minProb
        self.logFileName = logFileName
        self.verbose     = verbose
        self.logger      = getLogger( logFileName, verbose, 'ptc' )
        self.classes     = [ 'not_peak_or_trough', 'peak', 'trough' ]
        self.features    = [ 'feature1' ]
        self.dayDf       = None
        self.classifier  = None
        self.trnAccuracy = None
        self.oosAccuracy = None        
        self.normTrnMat  = None
        self.normOosMat  = None        
        
        assert method in [ 'bayes', 'gp', 'log' ], \
            'Method %s is not supported right now!' % method
        
        self.setDf()

    def setDf( self ):

        symbol     = self.symbol
        symFile    = self.symFile
        vixFile    = self.vixFile        
        thres      = self.ptThreshold
        nPTAvgDays = self.nPTAvgDays
        fileExt    = symFile.split( '.' )[-1]
        
        if fileExt == 'csv':
            dayDf = pd.read_csv( symFile ) 
        elif fileExt == 'pkl':
            dayDf = pd.read_pickle( symFile ) 
        else:
            assert False, 'Unknown input file extension %s' % fileExt
            
        assert symbol in dayDf.columns, 'Symbol %s not found in %s' \
            % ( symbol, symFile )

        dayDf[ 'Date' ] = \
            dayDf.Date.apply( lambda x : x.strftime( '%Y-%m-%d' ) )
        
        dayDf = dayDf.groupby( 'Date', as_index = False ).mean()
        
        if self.minVix is not None or self.maxVix is not None:

            fileExt = vixFile.split( '.' )[-1]
        
            if fileExt == 'csv':
                vixDf = pd.read_csv( vixFile ) 
            elif fileExt == 'pkl':
                vixDf = pd.read_pickle( vixFile ) 
            else:
                assert False, 'Unknown input file extension %s' % fileExt
            
            assert 'VIX' in vixDf.columns, 'VIX not found in %s' \
                % vixFile

            vixDf[ 'Date' ] = \
                vixDf.Date.apply( lambda x : x.strftime( '%Y-%m-%d' ) )
            
            vixDf = vixDf.groupby( 'Date', as_index = False ).mean()

            dayDf = dayDf.merge( vixDf, on = 'Date', how = 'left' )            
            dayDf = dayDf.interpolate( method = 'linear' )

        dayDf[ 'vel' ] = np.gradient( dayDf[ symbol ], 2 )
        dayDf[ 'acl' ] = np.gradient( dayDf[ 'vel' ], 2 )

        if nPTAvgDays is not None:
            dayDf[ 'avgBkd' ] = dayDf[ symbol ].\
                rolling( min_periods = 1,
                         window = nPTAvgDays ).mean()
            dayDf[ 'avgFwd' ] = dayDf[ symbol ].shift( -nPTAvgDays ).\
                rolling( min_periods = 1,
                         window = nAvgPTDays ).mean()

        dayDf[ 'feature1' ] = dayDf.acl 

        dayDf[ 'feature2' ] = dayDf[ symbol ] - \
            dayDf[ symbol ].shift(1) 
            
        dayDf = dayDf.dropna()
        
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

        if self.minVix is not None:
            dayDf = dayDf[ dayDf.VIX >= self.minVix ]
            
        if self.maxVix is not None:
            dayDf = dayDf[ dayDf.VIX <= self.maxVix ]

        dayDf[ 'Date' ] = dayDf.Date.astype( 'datetime64[ns]' )
        
        self.logger.info( 'A total of %d samples for %s! '
                          'Found %d peaks and %d troughs!',
                          dayDf.shape[0],
                          symbol,
                          dayDf[ dayDf.ptTag == PEAK ].shape[0],
                          dayDf[ dayDf.ptTag == TROUGH ].shape[0]  )

        self.dayDf = dayDf
            
    def plotDists( self ):

        df = self.dayDf
        
        sns.distplot( df[ df.ptTag == PEAK ][ 'feature1' ] )
        sns.distplot( df[ df.ptTag == TROUGH ][ 'feature1' ] )
        sns.distplot( df[ df.ptTag == NOT_PT ][ 'feature1' ] )
        
        plt.legend( [ 'Peak',
                      'Trough',
                      'Not Peak or Trough' ] )
        plt.xlabel( 'Feature1' )
        plt.ylabel( 'Distribution of ' + self.symbol )
        plt.show()

        if 'feature2' in self.features:
            sns.distplot( df[ df.ptTag == PEAK ][ 'feature2' ] )
            sns.distplot( df[ df.ptTag == TROUGH ][ 'feature2' ] )
            sns.distplot( df[ df.ptTag == NOT_PT ][ 'feature2' ] )
        
            plt.legend( [ 'Peak',
                          'Trough',
                          'Not Peak or Trough' ] )
            plt.xlabel( 'Feature2' )
            plt.ylabel( 'Distribution of ' + self.symbol )
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

    def plotScatter( self ):

        df = self.dayDf

        plt.scatter( df[ df.ptTag == NOT_PT ][ 'feature1' ],
                     df[ df.ptTag == NOT_PT ][ 'feature2' ],
                     color = 'orange', marker = 'o' )
        
        plt.scatter( df[ df.ptTag == PEAK ][ 'feature1' ],
                     df[ df.ptTag == PEAK ][ 'feature2' ],
                     c = 'red',
                     marker = 'o' )

        plt.scatter( df[ df.ptTag == TROUGH ][ 'feature1' ],
                     df[ df.ptTag == TROUGH ][ 'feature2' ],
                     c = 'blue',
                     marker = 's' )
        
        plt.legend( [ 'Not peak / trough', 'Peak', 'Trough' ] )
        plt.xlabel( 'feature1' )
        plt.ylabel( self.symbol )
        plt.show()
        
    def classify( self ):

        self.logger.info( 'Building a peak/trough classifier for %s',
                          self.symbol )
        
        df = self.dayDf
        X  = np.array( df[ self.features ] )
        y  = np.array( df.ptTag )

        if self.testRatio == 0:
            XTrn = X
            yTrn = y
        else:            
            XTrn, XOos, yTrn, yOos = train_test_split(
                X,
                y,
                test_size    = self.testRatio,
                random_state = 0
            )

        if self.method == 'bayes':
            obj = GaussianNB()
        elif self.method == 'gp':
            obj = GaussianProcessClassifier()
        elif self.method == 'log':
            obj = LogisticRegression()            
        else:
            assert False, 'Method %s not supported!' % self.method
            
        obj = obj.fit( XTrn, yTrn )
        
        self.classifier  = obj
        self.trnAccuracy = obj.score( XTrn, yTrn )
        self.logger.info( 'In-sample accuracy: %0.4f',
                          self.trnAccuracy )
        
        if self.testRatio != 0:
            self.oosAccuracy = obj.score( XOos, yOos )
            self.logger.info( 'Out-of-sample accuracy: %0.4f',
                              self.oosAccuracy )

        yTrnPred = obj.predict( XTrn )        
        confMat  = confusion_matrix( yTrn, yTrnPred )

        self.logger.info( 'In-sample confusion matrix:\n %s',
                          str( confMat ) )

        self.normTrnMat = self.normalizeConfMat( confMat )     
                
        self.logger.info( 'In-sample norm. confusion matrix:\n %s',
                          str( self.normTrnMat ) )

        if self.testRatio != 0:
            yOosPred = obj.predict( XOos )        
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
            self.dayDf[ self.features ].apply( self.getClass, axis = 1 )
        self.dayDf[ 'prdProb' ] = \
            self.dayDf[ self.features ].apply( self.getClassProb, axis = 1 )
        
    def getClass( self, vals ):

        X = np.array( [ vals ] )

        return self.classifier.predict( X )[0]

    def getClassProb( self, vals ):

        X     = np.array( [ vals ] )

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
