# ***********************************************************************
# Import libraries
# ***********************************************************************

import os, sys, dill
import datetime
import re
import random
import talib
import pickle
import numpy as np
import pandas as pd

sys.path.append( os.path.abspath( '../' ) )

import utl.utils as utl

from mod.mfdMod import MfdMod
from prt.prt import MfdPrt 

# ***********************************************************************
# Set inverse ETF hash
# ***********************************************************************

ETF_HASH = {  'TQQQ' : 'SQQQ',
              'SPY'  : 'SH',
              'DDM'  : 'DXD',
              'MVV'  : 'MZZ',
              'UWM'  : 'TWM',
              'SAA'  : 'SDD',
              'UYM'  : 'SMN',
              'UGE'  : 'SZK',
              'UCC'  : 'SCC',
              'FINU' : 'FINZ',
              'RXL'  : 'RXD',
              'UXI'  : 'SIJ',
              'URE'  : 'SRS',
              'ROM'  : 'REW',
              'UJB'  : 'SJB',
              'AGQ'  : 'ZSL',     
              'DIG'  : 'DUG',
              'USD'  : 'SSG',
              'ERX'  : 'ERY',
              'UYG'  : 'SKF',
              'UCO'  : 'SCO',
              'BOIL' : 'KOLD',
              'UPW'  : 'SDP',
              'UGL'  : 'GLL',
              'BIB'  : 'BIS',
              'UST'  : 'PST',
              'UBT'  : 'TBT' }
 
# ***********************************************************************
# Input
# ***********************************************************************

nSamples    = None

modDir      = 'models'
prtDir      = 'models'
modHead     = 'model_'
prtHead     = 'weights_'
pattern     = modHead + '\d+-\d+-\d+ \d+:\d+:\d+.dill'

jsonFlag    = False
outFile     = 'portfoio_perfs.csv'

# ***********************************************************************
# Get model files
# ***********************************************************************
        
modFiles = []
        
for fileName in os.listdir( modDir ):
                
    if not re.search( pattern, fileName ):
        continue
            
    modFiles.append( fileName )
                
if nSamples is not None:
    modFiles = random.sample( modFiles, nSamples )

# ***********************************************************************
# Evaluate
# ***********************************************************************

modFiles = sorted( modFiles )
outDf    = pd.DataFrame()

if jsonFlag:
    prtExt = '.json'
else:
    prtExt = '.pkl'
    
for modName in modFiles: 
            
    baseName = os.path.splitext( modName )[0]
    dateStr  = baseName.replace( modHead, '' )
    prtName  = prtHead + dateStr + prtExt
    modFile  = os.path.join( modDir, modName )            
    prtFile  = os.path.join( prtDir, prtName )

    if not os.path.exists( prtFile ):
        continue

    if jsonFlag:
        with open( prtFile, 'r' ) as fHd:
            wtHash = json.load( fHd )
    else:
        with open( prtFile, 'rb' ) as fHd:
            wtHash = list( pickle.load( fHd ).values() )[0]
    
    print( 'Processing %s ...' % modName )
    
    tmpDf = utl.evalMfdPrtPerf( modFile   = modFile,
                                 wtHash    = wtHash,
                                 shortFlag = False,
                                 invHash   = ETF_HASH   )

    outDf = pd.concat( [ outDf, tmpDf ] )

outDf.to_csv( outFile, index = False )

print( outDf.head( 20 ) )
