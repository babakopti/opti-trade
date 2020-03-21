import os, sys, time

sys.path.append( '../' )

from utl.utils import mergePiSymbols

from dat.assets import OPTION_ETFS as ETFS
from dat.assets import FUTURES, INDEXES

INDEXES  = INDEXES + [ 'VIX' ]

symbols = INDEXES + ETFS + FUTURES

print( 'Merging %s symbols' % len( symbols ) )

piDf = mergePiSymbols( symbols = symbols,
                       datDir  = '/Users/babak/workarea/data/pitrading_data',
                       minDate = '2010-01-01'   )

piDf.to_pickle( 'dfFile_pitrading_option.pkl' )
                     
