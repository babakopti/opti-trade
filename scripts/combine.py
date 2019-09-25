import os, sys, pickle
import numpy as np
import pandas as pd

sys.path.append( os.path.abspath( '../' ) )

modFiles = os.listdir( 'models' )

wtHash = {}

for item in modFiles:

    if item.split( '_' )[0] != 'weights':
        continue
    
    filePath = os.path.join( 'models', item )
    tmpHash = pickle.load( open( filePath, 'rb' ) )
    dateStr = list( tmpHash.keys() )[0]
    wtHash[ dateStr ] = tmpHash[ dateStr ]

print( wtHash )
