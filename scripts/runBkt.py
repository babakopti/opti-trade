# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os

sys.path.append( os.path.abspath( '../' ) )

from bkt.backtesters import MfdPrtBacktester

# ***********************************************************************
# Set some parameters and read data
# ***********************************************************************

futures = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

ETFs    = [ 'TQQQ', 'SPY', 'DDM', 'MVV', 'UWM',  'SAA',
            'UYM',  'UGE', 'UCC', 'FINU', 'RXL', 'UXI',
            'URE',  'ROM', 'UJB', 'AGQ',  'DIG', 'USD',
            'ERX',  'UYG', 'UCO', 'BOIL', 'UPW', 'UGL',
            'BIB', 'UST', 'UBT'  ]

# ***********************************************************************
# Run backtest
# ***********************************************************************

if __name__ == '__main__':
    bkt = MfdPrtBacktester( velNames   = ETFs + futures,
                            assets     = ETFs,                          
                            dfFile     = 'data/dfFile_kibot_2016plus.pkl',
                            bktBegDate = '2017-01-01 09:00:00',
                            bktEndDate = '2019-12-31 09:00:00',
                            sType      = 'ETF',
                            maxAssets  = 5,
                            nEvalDays  = 30,
                            nTrnDays   = 360,
                            nOosDays    = 3,
                            nPrdDays    = 1,
                            nMadDays    = 30,
                            maxOptItrs  = 500,
                            optTol      = 5.0e-2,
                            regCoef     = 5.0e-3,
                            factor      = 4.0e-05,
                            outBktFile  = 'portfolio.json',
                            modFlag     = True,
                            modHead     = 'model_',
                            prtHead     = 'weights_',
                            modDir      = 'models',
                            prtDir      = 'models',
                            logFileName = None,
                            verbose     = 1           )

    bkt.backtest()
    
    


