# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import numpy as np
import scipy as sp

from scipy.integrate import trapz

sys.path.append( os.path.abspath( '../' ) )

from ode.odeBase import OdeBaseConst

# ***********************************************************************
# OdeGeoConst: Geodesic ODE solver; 1st order; const. curv.
# ***********************************************************************

class OdeGeoConst( OdeBaseConst ):

    def fun( self, t, y ):

        nDims    = self.nDims
        Gamma    = self.Gamma
        timeInc  = self.timeInc
        nTimes   = self.nTimes
        srcTerm  = self.srcTerm
        srcVec   = np.zeros( shape = ( nDims ) , dtype = 'd' )
        tsId     = int( t / timeInc )

        if srcTerm is not None and tsId < nTimes:
            for m in range( nDims ):
                srcVec[m] = srcTerm[m][tsId]
                
        vals = -np.tensordot( Gamma,
                              np.tensordot( y, y, axes = 0 ),
                              ( ( 1, 2 ), ( 0, 1 ) ) )
        vals += srcVec
        
        return vals

    def jac( self, t, y ):
                       
        nDims    = self.nDims
        Gamma    = self.Gamma

        vals = -2.0 * np.tensordot( Gamma, y, axes = ( (2), (0) ) )

        return vals

# ***********************************************************************
# OdeAdjConst: Adjoint Geodesic solver; constant curvature
# ***********************************************************************

class OdeAdjConst( OdeBaseConst ):

    def fun( self, t, v ):

        nDims    = self.nDims
        Gamma    = self.Gamma
        timeInc  = self.timeInc
        nTimes   = self.nTimes
        actSol   = self.actSol
        adjSol   = self.adjSol
        varCoefs = self.varCoefs
        atnCoefs = self.atnCoefs

        vals     = np.zeros( shape = ( nDims ) , dtype = 'd' )
        tsId     = int( t / timeInc )

        assert tsId < nTimes, 'tsId should be smaller than nTimes!'

        adjVec  = np.zeros( shape = ( nDims ), dtype = 'd' )
        actVec  = np.zeros( shape = ( nDims ), dtype = 'd' )        

        for a in range( nDims ):
            adjVec[a] = adjSol[a][tsId]
            actVec[a] = actSol[a][tsId]

        vals = 2.0 * np.tensordot( Gamma,
                                   np.tensordot( v, adjVec, axes = 0 ),
                                   ( ( 0, 2 ), ( 0, 1 ) ) ) + \
                                   atnCoefs[tsId] * varCoefs * \
                                   ( adjVec - actVec )
        
        return vals

    def jac( self, t, v ):

        nDims   = self.nDims
        timeInc = self.timeInc
        nTimes  = self.nTimes
        Gamma   = self.Gamma
        adjSol  = self.adjSol

        vals    = np.zeros( shape = ( nDims, nDims ), dtype = 'd' )
        tsId    = int( t / timeInc )
 
        assert tsId < nTimes, 'tsId should be smaller than nTimes!'

        adjVec  = np.zeros( shape = ( nDims ), dtype = 'd' )

        for a in range( nDims ):
            adjVec[a] = adjSol[a][tsId]

        vals = 2.0 * np.tensordot( Gamma, adjVec, ( (2), (0) ) )
        vals = np.transpose( vals )

        return vals

# ***********************************************************************
# OdeGeoNN: Geodesic ODE solver; Geo-NN model
# ***********************************************************************

class OdeGeoNN(OdeBaseNN):

    def fun( self, t, y ):

        nDims    = self.nDims
        timeInc  = self.timeInc
        nTimes   = self.nTimes
        srcTerm  = self.srcTerm
        srcVec   = np.zeros( shape = ( nDims ) , dtype = 'd' )
        tsId     = int( t / timeInc )
        Gamma    = self.GammaFunc(tsId)

        if srcTerm is not None and tsId < nTimes:
            for m in range( nDims ):
                srcVec[m] = srcTerm[m][tsId]
                
        vals = -np.tensordot( Gamma,
                              np.tensordot( y, y, axes = 0 ),
                              ( ( 1, 2 ), ( 0, 1 ) ) )
        vals += srcVec
        
        return vals

    def jac( self, t, y ):
                       
        nDims    = self.nDims
        timeInc  = self.timeInc
        tsId     = int( t / timeInc )
        Gamma    = self.GammaFunc(tsId)

        vals = -2.0 * np.tensordot( Gamma, y, axes = ( (2), (0) ) )

        return vals

# ***********************************************************************
# OdeAdjNN: Adjoint Geodesic solver; Geo-NN model
# ***********************************************************************

class OdeAdjNN(OdeBaseNN):

    def fun( self, t, v ):

        nDims    = self.nDims
        timeInc  = self.timeInc
        nTimes   = self.nTimes
        actSol   = self.actSol
        adjSol   = self.adjSol
        varCoefs = self.varCoefs
        atnCoefs = self.atnCoefs

        tsId     = int( t / timeInc )
        Gamma    = self.GammaFunc(tsId)

        vals     = np.zeros( shape = ( nDims ) , dtype = 'd' )
        tsId     = int( t / timeInc )

        assert tsId < nTimes, 'tsId should be smaller than nTimes!'

        adjVec  = np.zeros( shape = ( nDims ), dtype = 'd' )
        actVec  = np.zeros( shape = ( nDims ), dtype = 'd' )        

        for a in range( nDims ):
            adjVec[a] = adjSol[a][tsId]
            actVec[a] = actSol[a][tsId]

        vals = 2.0 * np.tensordot( Gamma,
                                   np.tensordot( v, adjVec, axes = 0 ),
                                   ( ( 0, 2 ), ( 0, 1 ) ) ) + \
                                   atnCoefs[tsId] * varCoefs * \
                                   ( adjVec - actVec )
        
        return vals

    def jac( self, t, v ):

        nDims   = self.nDims
        timeInc = self.timeInc
        nTimes  = self.nTimes
        adjSol  = self.adjSol

        vals    = np.zeros( shape = ( nDims, nDims ), dtype = 'd' )
        tsId    = int( t / timeInc )
 
        assert tsId < nTimes, 'tsId should be smaller than nTimes!'

        Gamma    = self.GammaFunc(tsId)
        
        adjVec  = np.zeros( shape = ( nDims ), dtype = 'd' )

        for a in range( nDims ):
            adjVec[a] = adjSol[a][tsId]

        vals = 2.0 * np.tensordot( Gamma, adjVec, ( (2), (0) ) )
        vals = np.transpose( vals )

        return vals
    
