import os
import sys
import numpy as np

sys.path.append( '..' )

from ptc.ptc import PTClassifier

from dat.assets import SUB_ETF_HASH

symbols = list( SUB_ETF_HASH.keys() )

trnAccuracy = 0.0
oosAccuracy = 0.0
trnNormMat  = np.zeros( shape = ( 3, 3 ), dtype = 'd' )
oosNormMat  = np.zeros( shape = ( 3, 3 ), dtype = 'd' )

for symbol in symbols:
    
    dfFile = 'data/%s.pkl' % symbol
    
    ptcObj = PTClassifier( symbol      = symbol,
                           dfFile      = dfFile,
                           ptThreshold = 1.0e-2,
                           nAvgDays    = 7,
                           nPTAvgDays  = None,
                           testRatio   = 0.1,
                           method      = 'bayes',
                           logFileName = None,                    
                           verbose     = 1          )

    ptcObj.classify()

    trnAccuracy += ptcObj.getTrnMerits()[0]
    trnNormMat  += ptcObj.getTrnMerits()[1]
    oosAccuracy += ptcObj.getOosMerits()[0]
    oosNormMat  += ptcObj.getOosMerits()[1]
    
    ptcObj.save( 'pt_classifiers/ptc_%s.pkl' % symbol )
    #ptcObj.plotDists()
    #ptcObj.plotSymbol( actPeaks = True, prdPeaks = True )
    #ptcObj.plotSymbol( actTroughs = True, prdTroughs = True )

trnAccuracy /= len( symbols )
trnNormMat  /= len( symbols )
oosAccuracy /= len( symbols )
oosNormMat  /= len( symbols )

print( 'In-sample accuracy:', trnAccuracy ) 
print( 'Out-of-sample accuracy:', oosAccuracy )
print( 'In-sample norm. conf. mat.:', trnNormMat ) 
print( 'Out-of-sample norm. conf. mat.:', oosNormMat ) 
