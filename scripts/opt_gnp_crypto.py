from apply_gain_preservation import getGnpPerf

import json
import numpy as np
import pandas as pd

DF_FILE = 'data/dfFile_crypto.pkl'
PRT_FILE = 'portfolios/crypto_9PM_no_zcash_ptc_no_short.json'

prtWtsHash = json.load( open( PRT_FILE, 'r' ) )

perf_hash = {
    'std_coef': [],
    'pers_off': [],
    'num_pers': [],
    'mean_std_ratio': [],
}

for num_pers in [30, 60, 90, 120, 150]:
    for pers_off in range(1, 8):
        for std_coef in np.linspace( 1.0, 2.0, 11 ):
            perf_hash[ 'std_coef' ].append( std_coef )
            perf_hash[ 'pers_off' ].append( pers_off )
            perf_hash[ 'num_pers' ] .append( num_pers )
            perf_hash[ 'mean_std_ratio' ].append(
                getGnpPerf(std_coef, pers_off, num_pers, prtWtsHash, DF_FILE )
            )
        
perf_df = pd.DataFrame( perf_hash )

print( perf_df )

perf_df.to_csv( 'analysis-results/gnp_performance_crypto.csv', index = False )

