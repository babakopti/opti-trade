# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os

sys.path.append( os.path.abspath( '../' ) )

from mod.mfdMod import MfdMod

import dill
import numpy as np
import pandas as pd
import scipy as sp

# ***********************************************************************
# Some parameters
# ***********************************************************************

NULL  = 0
LONG  = 1
SHORT = 2

# ***********************************************************************
# Class MfdPrt: Model object for a manifold based portfolio
# ***********************************************************************

class MfdPrt:

    def __init__(   self,
                    modFile,
                    strategy     = 'mad_con_mfd',
                    begDate      = None,
                    endDate      = None,
                    minProbLong  = 0.75,
                    minProbShort = 0.75,
                    verbose      = 1          ):

        assert strategy == 'mad_con_mfd', 'Only mad_con_mfd is currently supported!'

        assert minProbLong > 0,  'minProbLong should be > 0!'
        assert minProbLong < 1,  'minProbLong should be < 1!'
        assert minProbShort > 0, 'minProbShort should be > 0!'
        assert minProbShort < 1, 'minProbShort should be < 1!'

        self.strategy     = strategy
        self.minProbLong  = minProbLong
        self.minProbShort = minProbShort
        self.verbose      = verbose

        self.mfdMod       = dill.load( open( modFile, 'rb' ) ) 
        self.ecoMfd       = self.mfdMod.ecoMfd

        if begDate is None:
            self.begDate = pd.to_datetime( self.ecoMfd.maxTrnDate )
        else:
            self.begDate = pd.to_datetime( begDate )

        if endDate is None:
            self.endDate = pd.to_datetime( self.ecoMfd.maxOosDate )
        else:
            self.endDate = pd.to_datetime( endDate )

        assert pd.to_datetime( self.ecoMfd.minTrnDate ) < self.begDate,\
            'minTrnDate should be before begDate!'

        self.assets       = self.ecoMfd.velNames
        self.retDf        = pd.DataFrame()
        
        self.setRetDf()
 
    def setRetDf( self ):

        ecoMfd = self.ecoMfd
        actSol = ecoMfd.actSol
        nTimes = ecoMfd.nTimes

        for m in range( ecoMfd.nDims ):
            asset     = ecoMfd.velNames[m]
            tmp       = ecoMfd.deNormHash[ velName ]
            slope     = tmp[0]
            intercept = tmp[1]
            df        = pd.DataFrame( { velName : slope * actSol[m][:nTimes] +\
                                            intercept } )

            self.retDf[ velName ] = np.log( df[ symbols ] ).pct_change().dropna()

    def getWeights( symbols, histFile, minDate = None ):

        assets    = self.assets
        numAssets = len( assets )
        guess     = np.ones( numAssets )
        cons      = { 'type' : 'eq', 'fun' : self.wtsSumCheck }

        results   = minimize( self.getMad, 
                              guess, 
                              constraints = cons      )
    
        weights   = results.x

        assert len( weights ) == numAssest,\
            'Inconsistent size of weights!'

        wtHash    = {}
        
        for i in range( numAssets ):
            wtHash[ assets[i] ] = weights[i]

        return wtHash
            
    def getMad( self, wts ):
        
        return ( self.retDf - self.retDf.mean() ).dot( wts ).abs().mean()

    def wtsSumCheck( self, x ):
        return sum( abs( x ) ) - 1
