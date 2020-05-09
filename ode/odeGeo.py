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
        srcCoefs = self.srcCoefs        
        timeInc  = self.timeInc
        nTimes   = self.nTimes
        srcTerm  = self.srcTerm
        src      = np.zeros( shape = ( nDims ) , dtype = 'd' )        
        tsId     = int( t / timeInc )

        if srcTerm is not None and tsId < nTimes:
            for m in range( nDims ):
                src[m] += srcTerm[m][tsId]

        if srcCoefs is not None:
            tmp = 0.0
            for k in range( len( srcCoefs ) ):
                tmp += srcCoefs[k] * ( t / nTimes )**k
                
            src += self.srcFct * tmp * y
                    
        vals = -np.tensordot( Gamma,
                              np.tensordot( y, y, axes = 0 ),
                              ( ( 1, 2 ), ( 0, 1 ) ) )
        vals += src
        
        return vals

    def jac( self, t, y ):
                       
        nDims    = self.nDims
        Gamma    = self.Gamma
        nTimes   = self.nTimes
        srcCoefs = self.srcCoefs
        src      = np.zeros( shape = ( nDims, nDims ) , dtype = 'd' )                

        if srcCoefs is not None:
            tmp = 0.0
            for k in range( len( srcCoefs ) ):
                tmp += srcCoefs[k] * ( t / nTimes )**k
            
            for m in range( nDims ):
                src[m][m] = self.srcFct * tmp
                
        vals = -2.0 * np.tensordot( Gamma, y, axes = ( (2), (0) ) ) + src

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
        srcCoefs = self.srcCoefs

        vals     = np.zeros( shape = ( nDims ) , dtype = 'd' )
        tsId     = int( t / timeInc )

        assert tsId < nTimes, 'tsId should be smaller than nTimes!'

        adjVec  = np.zeros( shape = ( nDims ), dtype = 'd' )
        actVec  = np.zeros( shape = ( nDims ), dtype = 'd' )        
        src     = np.zeros( shape = ( nDims ), dtype = 'd' )
        
        for a in range( nDims ):
            adjVec[a] = adjSol[a][tsId]
            actVec[a] = actSol[a][tsId]

        if srcCoefs is not None:
            tmp = 0.0
            for k in range( len( srcCoefs ) ):
                tmp += srcCoefs[k] * ( t / nTimes )**k
                
            src = -tmp * self.srcFct * v
            
        vals = 2.0 * np.tensordot( Gamma,
                                   np.tensordot( v, adjVec, axes = 0 ),
                                   ( ( 0, 2 ), ( 0, 1 ) ) ) + \
                                   atnCoefs[tsId] * varCoefs * \
                                   ( adjVec - actVec ) +\
                                   src
        
        return vals

    def jac( self, t, v ):

        nDims    = self.nDims
        timeInc  = self.timeInc
        nTimes   = self.nTimes
        Gamma    = self.Gamma
        srcCoefs = self.srcCoefs
        adjSol   = self.adjSol

        vals = np.zeros( shape = ( nDims, nDims ), dtype = 'd' )
        src  = np.zeros( shape = ( nDims, nDims ) , dtype = 'd' )                        
        tsId = int( t / timeInc )
 
        assert tsId < nTimes, 'tsId should be smaller than nTimes!'

        adjVec  = np.zeros( shape = ( nDims ), dtype = 'd' )

        for a in range( nDims ):
            adjVec[a] = adjSol[a][tsId]

        if srcCoefs is not None:
            tmp = 0.0
            for k in range( len( srcCoefs ) ):
                tmp += srcCoefs[k] * ( t / nTimes )**k
                
            for r in range( nDims ):
                src[r][r] = -tmp * self.srcFct
            
        vals = 2.0 * np.tensordot( Gamma, adjVec, ( (2), (0) ) ) + src
        vals = np.transpose( vals )

        return vals

