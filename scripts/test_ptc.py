import os
import sys
import joblib

sys.path.append( '..' )

from ptc.ptc import PTClassifier

ptcObj = PTClassifier( symbol      = 'ERX',
                       symFile     = 'data/ERX.pkl',
                       vixFile     = 'data/VIX.pkl',
                       ptThreshold = 1.0e-2,
                       nPTAvgDays  = None,
                       testRatio   = 0,
                       method      = 'bayes',
                       minVix      = None,
                       maxVix      = 60.0,
                       logFileName = None,                    
                       verbose     = 1          )

ptcObj.plotDists()
ptcObj.plotScatter()

ptcObj.classify()

ptcObj.plotSymbol( actPeaks = True, prdPeaks = True )

ptcObj.plotSymbol( actTroughs = True, prdTroughs = True )

#ptcObj.save('test.pkl')

