import sys
import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append( '../' )

import utl.utils as utl

prtFile = 'portfolios/crypto_9PM_raw_no_zcash.json'

outPrtFile = 'portfolios/crypto_9PM_no_zcash_no_short.json'

dfFile = 'data/dfFile_crypto.pkl'

prtWtsHash = json.load( open( prtFile, 'r' ) )

for date in prtWtsHash:
    
    for symbol in prtWtsHash[date]:
        prtWtsHash[date][symbol] = max(0.0, prtWtsHash[date][symbol])

    sumAbs = sum( [abs(x) for x in prtWtsHash[date].values()] )
    sumAbsInv = 1.0
    if sumAbs > 0:
        sumAbsInv = 1.0 / sumAbs

    for symbol in prtWtsHash[date]:
        prtWtsHash[date][symbol] = sumAbsInv * prtWtsHash[date][symbol]
        
json.dump( prtWtsHash, open( outPrtFile, 'w' ) )

