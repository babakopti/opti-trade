import os
import sys
import joblib

sys.path.append( '..' )

from ptc.ptc import PTClassifier

ptcObj = PTClassifier( symbol      = 'SPY',
                       dfFile      = 'data/SPY.pkl',
                       ptThreshold = 1.0e-3,
                       nAvgDays    = 7,
                       nPTAvgDays  = None,
                       testRatio   = 0.2,
                       method      = 'bayes',
                       logFileName = None,                    
                       verbose     = 1          )

#ptcObj.plotSymbol( actPeaks = True )
#ptcObj.plotSymbol( actTroughs = True )
ptcObj.plotDists()

ptcObj.classify()

ptcObj.plotSymbol( actPeaks = True, prdPeaks = True )
ptcObj.plotSymbol( actTroughs = True, prdTroughs = True )

#ptcObj.save('test.pkl')

