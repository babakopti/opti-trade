from apply_gain_preservation import getGnpPerf

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DF_FILE = 'data/dfFile_2020.pkl'
PRT_FILE = 'portfolios/nTrnDays_360_two_hours_ptc.json'

prtWtsHash = json.load( open( PRT_FILE, 'r' ) )

legends = []
X = np.linspace(1.0, 2.0, 11 )
perf_hash = {
    'std_coef': [],
    'pers_off': [],
    'mean_std_ratio': [],
}

for pers_off in range(1, 21):
    legends.append( 'pers_off %s' % str(pers_off) )
    Y = []
    for std_coef in X:
        Y.append(
            getGnpPerf(std_coef, pers_off, prtWtsHash, DF_FILE )
        )
        perf_hash[ 'std_coef' ].append( std_coef )
        perf_hash[ 'pers_off' ].append( pers_off )
        perf_hash [ 'mean_std_ratio' ].append( Y[-1] )
        
    plt.plot( X, Y )
    
plt.xlabel( 'std_coef' )
plt.ylabel( 'Performance (Mean/Std ratio)' )
plt.legend( legends )
plt.title( 'Tuning of GNP' )
plt.show()

perf_df = pd.DataFrame( perf_hash )

print( perf_df )

perf_df.to_csv( 'analysis-results/gnp_performance.csv', index = False )

