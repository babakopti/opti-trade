import os
import sys
import joblib

sys.path.append( '..' )

from ptc.ptc import PTClassifier

ptcObj = PTClassifier( symbol      = 'AGQ',
                       symFile     = 'data/AGQ.pkl',
                       vixFile     = 'data/VIX.pkl',
                       ptThreshold = 1.0e-2,
                       nAvgDays    = 7,
                       nPTAvgDays  = None,
                       testRatio   = 0.2,
                       method      = 'bayes',
                       minVix      = 40.0,
                       maxVix      = None,
                       logFileName = None,                    
                       verbose     = 1          )

ptcObj.plotDists()

ptcObj.classify()

ptcObj.plotSymbol( actPeaks = True, prdPeaks = True )
ptcObj.plotSymbol( actTroughs = True, prdTroughs = True )

ptcObj.plotScatter()

#ptcObj.save('test.pkl')

