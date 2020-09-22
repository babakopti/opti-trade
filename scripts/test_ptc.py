import os
import sys
import joblib

sys.path.append( '..' )

from ptc.ptc import PTClassifier

ptcObj = PTClassifier( symbol      = 'AGQ',
                       symFile     = 'data/AGQ.pkl',
                       vixFile     = 'data/VIX.pkl',
                       ptThreshold = 1.0e-2,
                       nPTAvgDays  = None,
                       testRatio   = 0.2,
                       method      = 'bayes',
                       minVix      = None,
                       maxVix      = 40.0,
                       logFileName = None,                    
                       verbose     = 1          )

ptcObj.plotDists()
ptcObj.plotScatter()

ptcObj.classify()

#ptcObj.plotSymbol( actPeaks = True, prdPeaks = True )
#ptcObj.plotSymbol( actTroughs = True, prdTroughs = True )

#ptcObj.save('test.pkl')

