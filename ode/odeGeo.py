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
# OdeGeoConst2: Geodesic ODE solver; 2nd order; const. curv.
# ***********************************************************************

class OdeGeoConst2( OdeBaseConst ):

    def fun( self, t, y ):

        nDims     = self.nDims
        Gamma     = self.Gamma
        beta      = self.beta
        timeInc   = self.timeInc
        nTimes    = self.nTimes
        actAvgSol = self.actAvgSol
        tsId      = int( t / timeInc )
        x         = y[:nDims]
        u         = y[nDims:]

        assert tsId < nTimes, 'tsId should be smaller than nTimes!'

        vec = -np.tensordot( Gamma,
                             np.tensordot( u, u, axes = 0 ),
                             ( ( 1, 2 ), ( 0, 1 ) ) ) - \
               beta * ( x - actAvgSol.transpose()[tsId] )

        vals = np.concatenate( [ u, vec ] )
        
        return vals

    def jac( self, t, y ):
                       
        nDims = self.nDims
        Gamma = self.Gamma
        beta  = self.beta
        u     = y[nDims:]

        vec11 = np.zeros( shape = ( nDims, nDims ), dtype = 'd' )
        vec12 = np.eye( nDims )        
        vec21 = -np.diag( beta )
        vec22 = -2.0 * np.tensordot( Gamma,
                                     u,
                                     axes = ( (2), (0) ) )

        vals  = np.block( [ [vec11, vec12], [vec21, vec22] ] )
        
        return vals

# ***********************************************************************
# OdeAdjConst2: Adjoint Geodesic solver; 2nd order; constant curvature
# ***********************************************************************

class OdeAdjConst2( OdeBaseConst ):

    def fun( self, t, y ):

        nDims     = self.nDims
        Gamma     = self.Gamma
        beta      = self.beta
        timeInc   = self.timeInc
        nTimes    = self.nTimes
        actSol    = self.actSol
        adjSol    = self.adjSol
        adjVel    = self.adjVel
        adjAcl    = self.adjAcl
        varCoefs  = self.varCoefs
        atnCoefs  = self.atnCoefs
        tsId      = int( t / timeInc )
        v         = y[:nDims]
        w         = y[nDims:]

        assert tsId < nTimes, 'tsId should be smaller than nTimes!'

        adjSolVec = adjSol.transpose()[tsId]
        adjVelVec = adjVel.transpose()[tsId]
        adjAclVec = adjAcl.transpose()[tsId]       
        actSolVec = actSol.transpose()[tsId]

        vec  = 2.0 * np.tensordot( Gamma,
                                   np.tensordot( w, adjVelVec, axes = 0 ),
                                   ( ( 0, 2 ), ( 0, 1 ) ) ) +\
               2.0 * np.tensordot( Gamma,
                                   np.tensordot( v, adjAclVec, axes = 0 ),
                                   ( ( 0, 2 ), ( 0, 1 ) ) ) - \
               beta * v - \
               atnCoefs[tsId] * varCoefs * ( adjSolVec - actSolVec )

        vals = np.concatenate( [ w, vec ] )
        
        return vals

    def jac( self, t, y ):

        nDims     = self.nDims
        timeInc   = self.timeInc
        nTimes    = self.nTimes
        Gamma     = self.Gamma
        adjVel    = self.adjVel
        adjAcl    = self.adjAcl
        tsId      = int( t / timeInc )        
 
        assert tsId < nTimes, 'tsId should be smaller than nTimes!'

        adjVelVec = adjVel.transpose()[tsId]
        adjAclVec = adjAcl.transpose()[tsId]

        vec11 = np.zeros( shape = ( nDims, nDims ), dtype = 'd' )
        vec12 = np.eye( nDims )        
        vec21 = 2.0 * np.tensordot( Gamma,
                                     adjAclVec,
                                     axes = ( (2), (0) ) ).transpose() - \
                np.diag( beta )
        
        vec22 = 2.0 * np.tensordot( Gamma,
                                     adjVelVec,
                                     axes = ( (2), (0) ) ).transpose()

        vals  = np.block( [ [vec11, vec12], [vec21, vec22] ] )        

        return vals
