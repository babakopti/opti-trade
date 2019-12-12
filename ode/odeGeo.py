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
        srcCoefs = self.srcCoefs
        srcTerm  = self.srcTerm
        vals     = np.zeros( shape = ( nDims ) , dtype = 'd' )
        srcVec   = np.zeros( shape = ( nDims ) , dtype = 'd' )

        tsId     = int( t / timeInc )

        if srcTerm is not None and tsId < nTimes:
            for m in range( nDims ):
                srcVec[m] = srcTerm[m][tsId]

        for m in range( nDims ):
            
            for a in range( nDims ):
                for b in range( nDims ):
                    vals[m]  = vals[m] - Gamma[m][a][b] * y[a] * y[b]

                vals[m] = vals[m] + srcCoefs[m][a] * y[a]

            vals[m] = vals[m] + srcCoefs[m][nDims]
            vals[m] = vals[m] + srcVec[m]
            
        return vals

    def jac( self, t, y ):
                       
        nDims    = self.nDims
        Gamma    = self.Gamma
        srcCoefs = self.srcCoefs
        vals     = np.zeros( shape = ( nDims, nDims ), dtype = 'd' )
        
        for m in range( nDims ):
            for l in range( nDims ):

                for q in range( nDims ):
                    vals[m][l]  = vals[m][l] - 2.0 * Gamma[m][l][q] * y[q] 

            vals[m][l] = vals[m][l] + srcCoefs[m][l]

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

        tmpVec1  = np.zeros( shape = ( nDims ), dtype = 'd' )
        tmpVec2  = np.zeros( shape = ( nDims ), dtype = 'd' )

        for a in range( nDims ):
            tmpVec1[a] = adjSol[a][tsId]
        
        for r in range( nDims ):
            for m in range( nDims ):
                tmpVec2[m] = np.dot( Gamma[m][r][:], tmpVec1 )
            vals[r]  = vals[r] +\
                       2.0 * np.dot( tmpVec2, v ) +\
                       varCoefs[r] * atnCoefs[tsId] *\
                       ( adjSol[r][tsId] - actSol[r][tsId] )

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

        tmpVec1  = np.zeros( shape = ( nDims ), dtype = 'd' )

        for a in range( nDims ):
            tmpVec1[a] = adjSol[a][tsId]

        for r in range( nDims ):
            for l in range( nDims ):
                vals[r][l]  = vals[r][l] +\
                    2.0 * np.dot( Gamma[l][r][:], tmpVec1 )

        return vals

