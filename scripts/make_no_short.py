import sys
import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append( '../' )

import utl.utils as utl

prtFile = 'portfolios/nTrnDays_360_two_hours_ptc.json'

outPrtFile = 'portfolios/nTrnDays_360_two_hours_ptc_no_short.json'

dfFile = 'data/dfFile_2020.pkl'

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

