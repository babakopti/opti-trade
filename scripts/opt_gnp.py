from apply_gain_preservation import getGnpPerf

import json
import numpy as np
import pandas as pd

DF_FILE = 'data/dfFile_2020.pkl'
PRT_FILE = 'portfolios/nTrnDays_360_two_hours_ptc.json'

prtWtsHash = json.load( open( PRT_FILE, 'r' ) )

perf_hash = {
    'std_coef': [],
    'pers_off': [],
    'num_pers': [],
    'mean_std_ratio': [],
}

for num_pers in [20, 40, 60, 80, 100, 120, 160, 200, 240]:
    for pers_off in [ 8 ]:
        for std_coef in [ 1.3 ]:
            perf_hash[ 'std_coef' ].append( std_coef )
            perf_hash[ 'pers_off' ].append( pers_off )
            perf_hash[ 'num_pers' ] .append( num_pers )
            perf_hash [ 'mean_std_ratio' ].append(
                getGnpPerf(std_coef, pers_off, num_pers, prtWtsHash, DF_FILE )
            )
        
perf_df = pd.DataFrame( perf_hash )

print( perf_df )

perf_df.to_csv( 'analysis-results/gnp_num_pers_test.csv', index = False )

