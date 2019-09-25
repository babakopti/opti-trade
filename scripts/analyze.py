import os, sys, dill
import numpy as np
import pandas as pd

sys.path.append( os.path.abspath( '../' ) )

modFiles = os.listdir( 'models' )

tmpList = []

for item in modFiles:

    if item.split( '_' )[0] != 'model':
        continue
    
    filePath = os.path.join( 'models', item )
    mfdMod = dill.load( open( filePath, 'rb' ) )
    ecoMfd = mfdMod.ecoMfd
    dateStr = item.split('_')[1][:10]
    dayName = pd.to_datetime( dateStr ).strftime('%A')
    cnt = ecoMfd.getOosTrendCnt()
    merit = ecoMfd.getMerit()
    oosMerit = ecoMfd.getOosMerit()
    print(  dateStr, cnt, merit, oosMerit )
    tmpList.append( cnt )


print( 'Mean = %.3f, min %.3f, max %.3f' %
       ( np.mean( tmpList ),
       min( tmpList ),
       max( tmpList ) ) )
