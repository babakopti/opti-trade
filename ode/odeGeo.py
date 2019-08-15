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

# ***********************************************************************
# OdeAdjConstOld: Old Adjoint Geodesic solver; constant curvature
# ***********************************************************************

class OdeAdjConstOld:

    def __init__( self,
                  Gamma,
                  actSol,
                  adjSol,
                  varCoefs,
                  nDims,
                  nSteps,
                  timeInc,
                  bkFlag   = False,
                  verbose  = 1           ):

        nTimes  = nSteps + 1

        assert Gamma.shape[0] == nDims,  'Incorrect Gamma size!'
        assert Gamma.shape[1] == nDims,  'Incorrect Gamma size!'
        assert Gamma.shape[2] == nDims,  'Incorrect Gamma size!'
        
        if actSol is not None:
            assert actSol.shape[0] == nDims,  'Incorrect actSol size!'
            assert actSol.shape[1] == nTimes, 'Incorrect actSol size!'

        if adjSol is not None:
            assert adjSol.shape[0] == nDims,  'Incorrect adjSol size!'
            assert adjSol.shape[1] == nTimes, 'Incorrect adjSol size!'

        assert len( varCoefs ) == nDims, 'Incorrect size for varCoefs!'

        self.Gamma    = Gamma
        self.actSol   = actSol
        self.adjSol   = adjSol
        self.varCoefs = varCoefs
        self.nDims    = nDims
        self.nSteps   = nSteps
        self.timeInc  = timeInc
        self.nTimes   = nTimes
        self.bkFlag   = bkFlag
        self.verbose  = verbose

        self.sol      = np.zeros( shape = ( nDims, nTimes ), dtype = 'd' )
        self.rhs      = np.zeros( shape = ( nDims ),         dtype = 'd' )
        self.lhs      = np.zeros( shape = ( nDims, nDims ),  dtype = 'd' )

    def solve( self ):

        nDims    = self.nDims        
        nSteps   = self.nSteps
        yPrev    = np.zeros( shape = ( nDims ), dtype = 'd' ) 

        linSolveTime = 0
        rhsTime      = 0
        lhsTime      = 0
        for stepId in range( nSteps ):
            
            t0 = time.time()
            self.setRhs( yPrev, stepId )
            rhsTime += time.time() - t0

            t0 = time.time()
            self.setLhs( stepId )
            lhsTime += time.time() - t0

            try:
                t0    = time.time()
                yCurr = np.linalg.solve( self.lhs, self.rhs )
                linSolveTime += time.time() - t0
            except:
                return False
        
            tsId = stepId + 1

            for m in range( nDims ):
                self.sol[m][tsId] = yCurr[m]

            yPrev = yCurr.copy()

        if self.verbose > 1:
            print( 'Linear solver for adjoint took', linSolveTime, 'seconds.' )
            print( 'LHS formation for adjoint took', lhsTime, 'seconds.' )
            print( 'RHS formation for adjoint took', rhsTime, 'seconds.' )

        return True

    def setRhs( self, yPrev, stepId ):

        nDims   = self.nDims
        nSteps  = self.nSteps
        timeInc = self.timeInc
        actSol  = self.actSol
        adjSol  = self.adjSol
        coefs   = self.varCoefs

        assert len( yPrev ) == nDims, 'Incorrect array size!'
        assert stepId < nSteps, 'Incorrect stepId!'
        
        tsId = stepId + 1 

        if self.bkFlag:
            fct = -1.0
        else:
            fct = 1.0

        for r in range( nDims ):
            self.rhs[r] = yPrev[r] +\
                fct * timeInc * coefs[r] * ( adjSol[r][tsId] - actSol[r][tsId] ) 

    def setLhs( self, stepId ):

        nDims   = self.nDims
        nSteps  = self.nSteps
        timeInc = self.timeInc
        Gamma   = self.Gamma
        adjSol  = self.adjSol
        
        assert stepId < nSteps, 'Incorrect stepId!'

        tsId = stepId + 1

        if self.bkFlag:
            fct = -1.0
        else:
            fct = 1.0

        tmpVec1 = np.zeros( shape = ( nDims ), dtype = 'd' )

        for a in range( nDims ):
            tmpVec1[a] = -2.0 * fct * timeInc * adjSol[a][tsId]
            
        for r in range( nDims ):
            for m in range( nDims ):
                tmpVec2       = Gamma[m][r][:]
                self.lhs[r,m] = np.dot( tmpVec1, tmpVec2 )

            self.lhs[r][r] += 1.0

    def getSol( self ):

        return self.sol
