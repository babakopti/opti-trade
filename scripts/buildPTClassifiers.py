import os
import sys

sys.path.append( '..' )

from ptc.ptc import PTClassifier

from dat.assets import SUB_ETF_HASH

symbols = list( SUB_ETF_HASH.keys() )

for symbol in symbols:
    
    dfFile = '/var/data/%s.pkl' % symbol
    
    ptcObj = PTClassifier( symbol      = symbol,
                           dfFile      = dfFile,
                           ptThreshold = 1.0e-3,
                           nAvgDays    = 7,
                           nPTAvgDays  = None,
                           testRatio   = 0.2,
                           method      = 'bayes',
                           logFileName = None,                    
                           verbose     = 1          )

    ptcObj.classify()
    ptcObj.save( 'ptc_%s.pkl' % symbol )

