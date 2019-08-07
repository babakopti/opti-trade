# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os

sys.path.append( os.path.abspath( '../' ) )

import time
import numpy as np
import scipy as sp
from scipy.integrate import trapz

from ode.odeBase import OdeBase 
from ode.odeBase import OdeBaseExp
from ode.odeBase import OdeBaseConst
from ode.odeBase import OdeBaseLin
from ode.odeBase import OdeBaseDLin
from ode.odeBase import OdeBaseDQuad

# ***********************************************************************
# OdeGeoExp: Geodesic ODE solver; exp. metric
# ***********************************************************************

class OdeGeoExp( OdeBaseExp ):

    def fun( self, t, y ):

        nDims      = self.nDims
        GammaCoefs = self.GammaCoefs
        vals       = np.zeros( shape = ( 2 * nDims ) , dtype = 'd' )

        for m in range( nDims ):

            vals[m] = y[m + nDims]

            for a in range( nDims ):
                tmpVec          = GammaCoefs[a][:] - GammaCoefs[m][:]
                tmpVal          = np.dot( tmpVec, y[:nDims] )
                expTerm         = np.exp( tmpVal )
                vals[m + nDims] = vals[m + nDims] +\
                            0.5 * GammaCoefs[a][m] * expTerm * ( y[a + nDims] )**2 -\
                            GammaCoefs[m][a] * y[a + nDims] * y[m + nDims]

        return vals

    def jac( self, t, y ):

        nDims      = self.nDims
        GammaCoefs = self.GammaCoefs
        vals       = np.zeros( shape = ( 2 * nDims, 2 * nDims ), dtype = 'd' )
        delta      = lambda a, b : 1.0 if a == b else 0.0

        for m in range( nDims ):
            for p in range( nDims ):

                vals[m, p]         = 0.0
                vals[m, p + nDims] = delta(m,p)
                
                for a in range( nDims ):
                    tmpVec             = GammaCoefs[a][:] - GammaCoefs[m][:]
                    tmpVal             = np.dot( tmpVec, y[:nDims] )
                    expTerm            = np.exp( tmpVal )
                    tmp1               = GammaCoefs[a][p] - GammaCoefs[m][p]
                    vals[m + nDims, p] = vals[m + nDims, p] +\
                        0.5 * GammaCoefs[a][m] * tmp1 * expTerm * ( y[a + nDims] )**2

                tmpVec   = GammaCoefs[p][:] - GammaCoefs[m][:]
                tmpVal   = np.dot( tmpVec, y[:nDims] )
                expTerm  = np.exp( tmpVal )

                vals[m + nDims, p + nDims] = vals[m + nDims, p + nDims] +\
                    GammaCoefs[p][m] * expTerm  * y[p + nDims] -\
                    GammaCoefs[m][p] * y[m + nDims]

                for a in range( nDims ):
                    vals[m + nDims, p + nDims] = vals[m + nDims, p + nDims] -\
                        GammaCoefs[m][a] * y[a + nDims] * delta(m,p)

        return vals

# ***********************************************************************
# OdeAdjExp: Adjoint Geodesic solver; exp metric
# ***********************************************************************

class OdeAdjExp:

    def __init__( self,
                  GammaCoefs,
                  actSol,
                  adjObj,
                  nDims,
                  nSteps,
                  timeInc,
                  verbose  = 1           ):

        nTimes  = nSteps + 1

        assert GammaCoefs.shape[0] == nDims, 'Incorrect GammaCoefs size!'
        assert GammaCoefs.shape[1] == nDims, 'Incorrect GammaCoefs size!'
        
        assert actSol.shape[0] == nDims,  'Incorrect actSol size!'
        assert actSol.shape[1] == nTimes, 'Incorrect actSol size!'

        adjSol = adjObj.getSol()
        assert adjSol.shape[0] == nDims,  'Incorrect adjSol size!'
        assert adjSol.shape[1] == nTimes, 'Incorrect adjSol size!'

        adjVel = adjObj.getVel()
        assert adjVel.shape[0] == nDims,  'Incorrect adjVel size!'
        assert adjVel.shape[1] == nTimes, 'Incorrect adjVel size!'

        adjAcl = adjObj.getAcl()
        assert adjAcl.shape[0] == nDims,  'Incorrect adjAcl size!'
        assert adjAcl.shape[1] == nTimes, 'Incorrect adjAcl size!'

        self.GammaCoefs = GammaCoefs
        self.actSol     = actSol
        self.adjSol     = adjSol
        self.adjVel     = adjVel
        self.adjAcl     = adjAcl
        self.nDims      = nDims
        self.nSteps     = nSteps
        self.timeInc    = timeInc
        self.nTimes     = nTimes
        self.verbose    = verbose

        self.sol        = np.zeros( shape = ( nDims, nTimes ), dtype = 'd' )
        self.vel        = np.zeros( shape = ( nDims, nTimes ), dtype = 'd' )
        self.acl        = np.zeros( shape = ( nDims, nTimes ), dtype = 'd' )
        self.rhs        = np.zeros( shape = ( 2 * nDims ),     dtype = 'd' )
        self.lhs        = np.zeros( shape = ( 2 * nDims, 2 * nDims ),  
                                    dtype = 'd' )

    def solve( self ):

        nDims    = self.nDims        
        nSteps   = self.nSteps
        yPrev    = np.zeros( shape = ( 2 * nDims ), dtype = 'd' ) 

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
        
            self.setSol( yCurr, stepId )

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

        assert len( yPrev ) == 2 * nDims, 'Incorrect array size!'
        assert stepId < nSteps, 'Incorrect stepId!'
        
        tsId = stepId + 1 

        for l in range( nDims ):
            self.rhs[l] = yPrev[l] 

            self.rhs[l + nDims] = yPrev[l + nDims] -\
                timeInc * ( adjSol[l][tsId] - actSol[l][tsId] ) 

    def setLhs( self, stepId ):

        nDims      = self.nDims
        nSteps     = self.nSteps
        timeInc    = self.timeInc
        GammaCoefs = self.GammaCoefs
        adjSol     = self.adjSol
        adjVel     = self.adjVel
        adjAcl     = self.adjAcl
        delta      = lambda a, b : 1.0 if a == b else 0.0

        assert stepId < nSteps, 'Incorrect stepId!'

        tsId = stepId + 1

        tmpSol = np.zeros( shape = ( nDims ), dtype = 'd' )
        tmpVel = np.zeros( shape = ( nDims ), dtype = 'd' )
        tmpAcl = np.zeros( shape = ( nDims ), dtype = 'd' )

        for a in range( nDims ):
            tmpSol[a] = adjSol[a][tsId]
            tmpVel[a] = adjVel[a][tsId]
            tmpAcl[a] = adjAcl[a][tsId]

        for l in range( nDims ):
            for p in range( nDims ):
                tmpVec    = GammaCoefs[l][:] - GammaCoefs[p][:]
                tmpVal    = np.dot( tmpVec, tmpSol )
                expTermlp = np.exp( tmpVal )
                tmpVal    = np.dot( tmpVec, tmpVel )
                QTerm     = GammaCoefs[l][p] * expTermlp * tmpVal * tmpVel[l] +\
                    GammaCoefs[l][p] * expTermlp * tmpAcl[l] -\
                    GammaCoefs[p][l] * tmpAcl[p] -\
                    delta(p,l) * np.dot( GammaCoefs[p][:], tmpVel ) 

                for a in range( nDims ):
                    tmpVec     = GammaCoefs[a][:] - GammaCoefs[p][:]
                    tmpVal     = np.dot( tmpVec, tmpSol )
                    expTermap  = np.exp( tmpVal )
                    tmpVal     = expTermap * ( tmpVel[a] )**2
                    QTerm      = QTerm -\
                        0.5 * GammaCoefs[a][p] * GammaCoefs[a][l] * tmpVal +\
                        0.5 * GammaCoefs[a][p] * GammaCoefs[p][l] * tmpVal
 
                STerm        = GammaCoefs[l][p] * tmpVel[l] * expTermlp -\
                    GammaCoefs[p][l] * tmpVel[p] -\
                    delta(p,l) * np.dot( GammaCoefs[l][:], tmpVel )
 
                self.lhs[l][p] = delta(l,p)
                self.lhs[l][p + nDims] = -delta(l,p) * timeInc
                self.lhs[l + nDims][p] = QTerm * timeInc
                self.lhs[l + nDims][p + nDims] = delta(l,p) + STerm * timeInc 

    def setSol( self, y, stepId ):

        nDims  = self.nDims
        nSteps = self.nSteps

        assert len( y ) == 2 * nDims, 'Incorrect array size!'
        assert stepId < nSteps, 'Incorrect stepId!'

        tsId   = stepId + 1

        for m in range( nDims ):
            self.sol[m][tsId] = y[m]
            self.vel[m][tsId] = y[m + nDims]

    def getSol( self ):
        return self.sol

    def getVel( self ):
        return self.vel

    def getAcl( self ):
        return self.acl

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

class OdeAdjConst:

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

# ***********************************************************************
# OdeGeodesic: Geodesic ODE solver; 1st order
# ***********************************************************************

class OdeGeodesic( OdeBase ):

    def fun( self, t, y, stepId ):

        self.funCnt += 1

        t0      = time.time()
        nDims   = self.nDims
        Gamma   = self.Gamma
        tsId    = stepId + 1
        vals    = np.zeros( shape = ( nDims ) , dtype = 'd' )

        for m in range( nDims ):
            for a in range( nDims ):
                for b in range( nDims ):
                    vals[m]  = vals[m] - Gamma[m][a][b][tsId] * y[a] * y[b]

        if self.verbose > 2:
            print( 'Forming geodesic function took', 
                   int( time.time() - t0 ), 
                   'seconds!' )

        return vals

    def jac( self, t, y, stepId ):
                       
        self.jacCnt += 1

        t0    = time.time()
        nDims = self.nDims
        Gamma = self.Gamma
        tsId  = stepId + 1
        vals  = np.zeros( shape = ( nDims, nDims ), dtype = 'd' )
        
        for m in range( nDims ):
            for l in range( nDims ):
                for q in range( nDims ):
                    vals[m,l]  = vals[m,l] - 2.0 * Gamma[m][l][q][tsId] * y[q] 

        if self.verbose > 2:
            print( 'Forming geodesic jacobian took', 
                   int( time.time() - t0 ), 
                   'seconds!' )

        return vals

# ***********************************************************************
# OdeAdjGeodesic: Adjoint Geodesic ODE solver; 1st order
# ***********************************************************************

class OdeAdjGeodesic( OdeBase ):

    def fun( self, t, v, stepId ):

        self.funCnt += 1

        t0      = time.time()
        nDims   = self.nDims
        nTimes  = self.nTimes
        timeInc = self.timeInc
        Gamma   = self.Gamma
        actSol  = self.actSol
        adjSol  = self.adjSol
        tsId    = stepId + 1
        vals    = np.zeros( shape = ( nDims ) , dtype = 'd' )

        for r in range( nDims ):
            for m in range( nDims ):
                for a in range( nDims ):
                    vals[r]  = vals[r] +\
                        2.0 * Gamma[m][a][r][tsId] * adjSol[a][tsId] * v[m]

            vals[r] += ( adjSol[r][tsId] - actSol[r][tsId] )

        if self.verbose > 2:
            print( 'Forming adjoint function took', 
                   int( time.time() - t0 ), 
                   'seconds!' )
                        
        return vals

    def jac( self, t, v, stepId ):

        self.jacCnt += 1

        t0      = time.time()
        nDims   = self.nDims
        timeInc = self.timeInc
        Gamma   = self.Gamma
        adjSol  = self.adjSol
        tsId    = stepId + 1
        vals    = np.zeros( shape = ( nDims, nDims ), dtype = 'd' )
        delta   = lambda r, l: 1.0 if r == l else 0.0
 
        for r in range( nDims ):
            for l in range( nDims ):
                for a in range( nDims ):
                    vals[r,l]  = vals[r,l] +\
                        2.0 * Gamma[l][a][r][tsId] * adjSol[a][tsId] 

        if self.verbose > 2:
            print( 'Forming adjoint jacobian took', 
                   int( time.time() - t0 ), 
                   'seconds!' )

        return vals

# ***********************************************************************
# OdeGeoLin: Geodesic ODE solver; 2nd order
# ***********************************************************************

class OdeGeoLin( OdeBaseLin ):

    def fun( self, t, y, stepId ):

        nDims      = self.nDims
        GammaCoefs = self.GammaCoefs
        vals       = np.zeros( shape = ( 2 * nDims ) , dtype = 'd' )

        for m in range( nDims ):

            vals[ m ]  = y[ m + nDims ]

            for a in range( nDims ):
                for b in range( nDims ):
                    for s in range( nDims ):
                        vals[ m + nDims ]  = vals[ m + nDims ] -\
                            GammaCoefs[m][a][b][s] *\
                            y[ a + nDims ] *\
                            y[ b + nDims ] *\
                            y[ s ]

                    vals[ m + nDims ]  = vals[ m + nDims ] -\
                        GammaCoefs[m][a][b][nDims] *\
                        y[ a + nDims ] *\
                        y[ b + nDims ]

        return vals

    def jac( self, t, y, stepId ):

        nDims      = self.nDims
        GammaCoefs = self.GammaCoefs
        vals       = np.zeros( shape = ( 2 * nDims, 2 * nDims ), dtype = 'd' )

        for m in range( nDims ):
            for l in range( nDims ):

                vals[ m, l ]         = 0.0
                vals[ m, l + nDims ] = 1.0

                for a in range( nDims ):
                    for b in range( nDims ):
                        vals[ m + nDims, l ]  = vals[ m + nDims, l ] - \
                            GammaCoefs[m][a][b][l] *\
                            y[ a + nDims ] *\
                            y[ b + nDims ]
                    for s in range( nDims ):
                        vals[ m + nDims, l + nDims ]  = vals[ m + nDims, l + nDims ] -\
                            2.0 * GammaCoefs[m][a][l][s] *\
                            y[ a + nDims ] *\
                            y[ s ] 

                    vals[ m + nDims, l + nDims ]  = vals[ m + nDims, l + nDims ] -\
                        2.0 * GammaCoefs[m][a][l][nDims] *\
                        y[ a + nDims ]

        return vals

# ***********************************************************************
# OdeAdj: Adjoint Geodesic ODE solver; 2nd order
# ***********************************************************************

class OdeAdjLin( OdeBaseLin ):

    def fun( self, t, v, stepId ):

        nDims      = self.nDims
        nTimes     = self.nTimes
        timeInc    = self.timeInc
        GammaCoefs = self.GammaCoefs
        actSol     = self.actSol
        actVel     = self.actVel
        actAcl     = self.actAcl
        adjObj     = self.adjObj
        tsId       = stepId + 1
        vals       = np.zeros( shape = ( 2 * nDims ) , dtype = 'd' )

        adjSol     = adjObj.getSol()
        adjVel     = adjObj.getVel()
        adjAcl     = adjObj.getAcl()

        for i in range( nDims ):

            vals[ i ] = v[ i + nDims ]

            for m in range( nDims ):
                for b in range( nDims ):
                    for s in range( nDims ):
                        vals[ i + nDims ]  = vals[ i + nDims ] +\
                            2.0 * GammaCoefs[m][i][b][s] *\
                            adjSol[s][tsId] * adjVel[b][tsId] * v[ m + nDims ] +\
                            2.0 * GammaCoefs[m][i][b][s] *\
                            adjVel[s][tsId] * adjVel[b][tsId] * v[ m ] +\
                            2.0 * GammaCoefs[m][i][b][s] *\
                            adjSol[s][tsId] * adjAcl[b][tsId] * v[ m ] -\
                            GammaCoefs[m][s][b][i] *\
                            adjVel[s][tsId] * adjVel[b][tsId] * v[ m ]
                    vals[ i + nDims ]= vals[ i + nDims ] +\
                        2.0 * GammaCoefs[m][i][b][nDims] * adjAcl[b][tsId]

            vals[ i + nDims ] = vals[ i + nDims ] -\
                ( adjSol[i][tsId] - actSol[i][tsId] ) -\
                0*( adjAcl[i][tsId] - actAcl[i][tsId] )
                        
        return vals

    def jac( self, t, v, stepId ):

        nDims      = self.nDims
        timeInc    = self.timeInc
        GammaCoefs = self.GammaCoefs
        adjObj     = self.adjObj
        tsId       = stepId + 1
        vals       = np.zeros( shape = ( 2 * nDims, 2 * nDims ), dtype = 'd' )

        adjSol     = adjObj.getSol()
        adjVel     = adjObj.getVel()
        adjAcl     = adjObj.getAcl()
        
        for i in range( nDims ):
            for l in range( nDims ):
                vals[ i, l ]         = 0.0
                vals[ i, l + nDims ] = 1.0
                
                for b in range( nDims ):
                    for s in range( nDims ):
                        vals[ i + nDims, l ] = vals[ i + nDims, l ] +\
                            2.0 * GammaCoefs[l][i][b][s] *\
                            adjVel[s][tsId] * adjVel[b][tsId] +\
                            2.0 * GammaCoefs[l][i][b][s] *\
                            adjSol[s][tsId] * adjAcl[b][tsId] -\
                            GammaCoefs[l][s][b][i] *\
                            adjVel[s][tsId] * adjVel[b][tsId]

                        vals[ i + nDims, l + nDims ] = vals[ i + nDims, l + nDims ] +\
                            2.0 * GammaCoefs[l][i][b][s] *\
                            adjSol[s][tsId] * adjVel[b][tsId] 

        return vals

# ***********************************************************************
# OdeGeoDLin: Geodesic ODE solver; 2nd order; for discrete adjoint
# ***********************************************************************

class OdeGeoDLin( OdeBaseDLin ):

    def setInitSol( self ):

        self.yCurr = self.bcVec.copy()

        self.setSolVel( self.yCurr, 0 )

    def setRes( self ):
        
        nDims      = self.nDims
        GammaCoefs = self.GammaCoefs
        timeInc    = self.timeInc
        yCurr      = self.yCurr
        yPrev      = self.yPrev

        self.res.fill ( 0.0 )

        for m in range( nDims ):
            self.res[m] = yCurr[m] - yPrev[m] -\
                timeInc * yCurr[m + nDims]
            self.res[m + nDims] = yCurr[m + nDims] - yPrev[m + nDims]
            for a in range( nDims ):
                for b in range( nDims ):
                    for s in range( nDims ):
                        self.res[m + nDims] += timeInc * GammaCoefs[m][a][b][s] *\
                            yCurr[a + nDims] * yCurr[b + nDims] * yCurr[s]

                    self.res[m + nDims] += timeInc * GammaCoefs[m][a][b][nDims] *\
                        yCurr[a + nDims] * yCurr[b + nDims]
        return

    def setLhs( self ):
        
        nDims      = self.nDims
        GammaCoefs = self.GammaCoefs
        timeInc    = self.timeInc
        yCurr      = self.yCurr
        delta      = lambda a,b: 1 if a == b else 0

        self.lhs.fill( 0.0 )

        for m in range( nDims ):
            for l in range( nDims ):

                self.lhs[m][l]                 = delta(m,l)
                self.lhs[m][l + nDims]         = -timeInc * delta(m,l)
                self.lhs[m + nDims][l + nDims] = delta(m,l)

                for a in range( nDims ):
                    for b in range( nDims ):

                        self.lhs[m + nDims][l] += timeInc * GammaCoefs[m][a][b][l] *\
                            yCurr[a + nDims] * yCurr[b + nDims]

                        self.lhs[m + nDims][l + nDims] += 2.0 * timeInc * GammaCoefs[m][a][l][b] *\
                            yCurr[a + nDims] * yCurr[b] 

                    self.lhs[m + nDims][l + nDims] += 2.0 * timeInc * GammaCoefs[m][a][l][nDims] *\
                        yCurr[a + nDims] 

        return

    def solve( self ):

        nSteps  = self.nSteps
        verbose = self.verbose
        
        self.setInitSol()

        if self.verbose > 0:
            print( '\nSolving geodesic equation...\n' )

        for stepId in range( nSteps ):
            
            if verbose > 1:
                print( 'Solving step', stepId + 1 )
                
            sFlag = self.solveStep()
            
            if not sFlag:
                break

            if verbose > 1:
                print( 'Current solution :', self.yCurr )

            self.setSolVel( self.yCurr, stepId + 1 )

        return sFlag 
    
    def solveStep( self ):

        self.yPrev = self.yCurr.copy()

        convFlag   = False
        
        for itr in range( self.nMaxItrs ):

            self.setRes()
            self.setLhs()

            res     = self.res
            lhs     = self.lhs
            
            delSol  = np.linalg.solve( lhs, -res )

            self.yCurr = self.yCurr + delSol

            errRes = np.linalg.norm( res ) 
            errDel = np.linalg.norm( delSol ) 
            err    = max( errRes, errDel )

            if self.verbose > 1:
                print( 'err =', err )

            if  err < self.tol:
                convFlag = True
                break
            
        return convFlag

# ***********************************************************************
# OdeAdjDLin: Adjoint ODE solver; 2nd order; for discrete adjoint
# ***********************************************************************

class OdeAdjDLin( OdeBaseDLin ):

    def setRhs( self, tsId ):

        nDims   = self.nDims
        nTimes  = self.nTimes
        actSol  = self.actSol
        actVel  = self.actVel
        adjSol  = self.adjSol
        adjVel  = self.adjVel
        yCurr   = self.yCurr
        yPrev   = self.yPrev
        wts     = self.varWeights

        assert len( yPrev ) == 2 * nDims, 'Incorrect array size!'
        assert tsId < nTimes, 'Incorrect tsId!'

        self.res.fill ( 0.0 )

        for l in range( nDims ):
            self.res[l] = yPrev[l] +\
                self.solFuncFct *\
                wts[l] * ( actSol[l][tsId] - adjSol[l][tsId] )
            self.res[l + nDims] = yPrev[l + nDims] +\
                self.velFuncFct *\
                wts[l] * ( actVel[l][tsId] - adjVel[l][tsId] )

    def setLhs( self, tsId ):

        nDims      = self.nDims
        nTimes     = self.nTimes
        timeInc    = self.timeInc
        GammaCoefs = self.GammaCoefs
        adjSol     = self.adjSol
        adjVel     = self.adjVel
        yCurr      = self.yCurr
        yPrev      = self.yPrev
        delta      = lambda a,b: 1 if a == b else 0
        
        assert tsId < nTimes, 'Incorrect tsId!'

        self.lhs.fill (  0.0 )

        for l in range( nDims ):
            for q in range( nDims ):

                self.lhs[l][q]                 = delta(l,q)
                self.lhs[l + nDims][q]         = -timeInc * delta(l,q)
                self.lhs[l + nDims][q + nDims] = delta(l,q)
                for a in range( nDims ):
                    for b in range( nDims ):
                        self.lhs[l][q + nDims] += timeInc * GammaCoefs[q][a][b][l] *\
                            adjVel[a][tsId] * adjVel[b][tsId]

                        self.lhs[l + nDims][q + nDims] += 2.0 * timeInc * GammaCoefs[q][a][l][b] *\
                            adjVel[a][tsId] * adjSol[b][tsId] 
                
                    self.lhs[l + nDims][q + nDims] += 2.0 * timeInc * GammaCoefs[q][a][l][nDims] *\
                        adjVel[a][tsId]

    def solve( self ):

        nSteps   = self.nSteps

        if self.verbose > 0:
            print( '\nSolving adjoint geodesic equation...\n' )

        for stepId in range( nSteps-1, -1, -1 ):
            
            self.yPrev = self.yCurr.copy()

            self.setRhs( stepId + 1 )
            self.setLhs( stepId + 1 )
            
            try:
                self.yCurr = np.linalg.solve( self.lhs, self.res )
            except:
                return False
        
            self.setSolVel( self.yCurr, stepId + 1 )

        return True

# ***********************************************************************
# OdeLinBvp: Geodesic/adjoint BVP solver; 2nd order
# ***********************************************************************

class OdeLinBvp:

    def __init__( self,
                  GammaCoefs,
                  initSol,
                  endSol,
                  timeInc,
                  nSteps,
                  eqnType,
                  intgType = 'vode',
                  initVel  = None,
                  actSol   = None,
                  actVel   = None,
                  actAcl   = None,
                  adjObj   = None,
                  tol      = 1.0e-4,
                  nMaxItrs = 20,
                  verbose  = 1           ):

        assert len( initSol ) == len( endSol ), 'Inconsistent lengths!'

        assert eqnType in [ 'geodesic', 'adjoint' ], 'Unkown equation!'
 
        nDims   = len( initSol )
        nTimes  = nSteps + 1
        nTmp    = nDims  + 1

        assert GammaCoefs.shape[0] == nDims, 'Incorrect Gamma size!'
        assert GammaCoefs.shape[1] == nDims, 'Incorrect Gamma size!'
        assert GammaCoefs.shape[2] == nDims, 'Incorrect Gamma size!'
        assert GammaCoefs.shape[3] == nTmp,  'Incorrect Gamma size!'
        
        if initVel is not None:
            assert len( initVel ) == nDims, 'Icorrect initVel size!'

        if actSol is not None:
            assert actSol.shape[0] == nDims,  'Incorrect actSol size!'
            assert actSol.shape[1] == nTimes, 'Incorrect actSol size!'

        if actVel is not None:
            assert actVel.shape[0] == nDims,  'Incorrect actVel size!'
            assert actVel.shape[1] == nTimes, 'Incorrect actVel size!'

        if actAcl is not None:
            assert actAcl.shape[0] == nDims,  'Incorrect actAcl size!'
            assert actAcl.shape[1] == nTimes, 'Incorrect actAcl size!'

        self.GammaCoefs = GammaCoefs
        self.initSol    = initSol
        self.endSol     = endSol
        self.nDims      = nDims
        self.timeInc    = timeInc
        self.nSteps     = nSteps
        self.eqnType    = eqnType
        self.intgType   = intgType
        self.initVel    = initVel
        self.actSol     = actSol
        self.actVel     = actVel
        self.actAcl     = actAcl
        self.adjObj     = adjObj
        self.tol        = tol
        self.nMaxItrs   = nMaxItrs
        self.verbose    = verbose
        self.nTimes     = nTimes
        self.odeObj     = None

        if self.initVel is None:
            self.initVel = np.zeros( shape = ( nDims ), dtype = 'd' )

    def fun( self, vel ):

        nDims   = self.nDims
        nSteps  = self.nSteps
        nTimes  = self.nTimes
        timeInc = self.timeInc
        eqnType = self.eqnType
        bcVec   = np.zeros( shape = ( 2 * nDims ), dtype = 'd' )
        resVec  = np.zeros( shape = (     nDims ), dtype = 'd' )

        assert len( vel ) == nDims, 'Incorrect size!'

        for m in range( nDims ):
            bcVec[m]         = self.initSol[m]
            bcVec[m + nDims] = vel[m] 
            
        if eqnType == 'geodesic':

            odeObj  = OdeGeoLin( GammaCoefs = self.GammaCoefs,
                                 bcVec      = bcVec,
                                 bcTime     = 0.0,
                                 timeInc    = timeInc,
                                 nSteps     = nSteps,
                                 intgType   = 'vode',
                                 tol        = 1.0e-8,
                                 nMaxItrs   = 1000,
                                 verbose    = 0  )

        elif eqnType == 'adjoint':

            odeObj  = OdeAdjLin( GammaCoefs = self.GammaCoefs,
                                 bcVec      = bcVec,
                                 bcTime     = 0.0,
                                 timeInc    = timeInc,
                                 nSteps     = nSteps,
                                 intgType   = 'dopri5',
                                 actSol     = self.actSol,
                                 actVel     = self.actVel,
                                 actAcl     = self.actAcl,
                                 adjObj     = self.adjObj,
                                 tol        = 1.0e-8,
                                 nMaxItrs   = 1000,
                                 verbose    = 0  )

        sFlag   = odeObj.solve()

        if not sFlag:
            print( eqnType, 'equation did not converge!' )
            sys.exit()
            return None

        sol     = odeObj.getSol()

        for m in range( nDims ):
            resVec[m] = sol[m][nTimes-1] - self.endSol[m]

        return resVec

    def solve( self ):
        
        nDims   = self.nDims
        nSteps  = self.nSteps
        timeInc = self.timeInc
        eqnType = self.eqnType
        bcVec   = np.zeros( shape = ( 2 * nDims ), dtype = 'd' )

        optObj  = sp.optimize.least_squares( fun = self.fun, 
                                             x0  = self.initVel )

        sFlag  = optObj.success

        if not sFlag:
            return False

        for m in range( nDims ):
            bcVec[m]         = self.initSol[m]
            bcVec[m + nDims] = optObj.x[m]

        if eqnType == 'geodesic':

            odeObj  = OdeGeoLin( GammaCoefs = self.GammaCoefs,
                                 bcVec      = bcVec,
                                 bcTime     = 0.0,
                                 timeInc    = timeInc,
                                 nSteps     = nSteps,
                                 intgType   = 'vode',
                                 tol        = 1.0e-8,
                                 nMaxItrs   = 1000,
                                 verbose    = self.verbose  )

        elif eqnType == 'adjoint':

            odeObj  = OdeAdjLin( GammaCoefs = self.GammaCoefs,
                                 bcVec      = bcVec,
                                 bcTime     = 0.0,
                                 timeInc    = timeInc,
                                 nSteps     = nSteps,
                                 intgType   = 'dopri5',
                                 actSol     = self.actSol,
                                 actVel     = self.actVel,
                                 actAcl     = self.actAcl,
                                 adjObj     = self.adjObj,
                                 tol        = 1.0e-8,
                                 nMaxItrs   = 1000,
                                 verbose    = self.verbose  )

        sFlag   = odeObj.solve()

        self.odeObj = odeObj

        return sFlag 

    def getSol( self ):
        return self.odeObj.sol

    def getVel( self ):
        return self.odeObj.vel

    def getAcl( self ):
        return self.odeObj.acl

# ***********************************************************************
# OdeGeoDQuad: Geodesic ODE solver; 2nd order; for discrete adjoint
# ***********************************************************************

class OdeGeoDQuad( OdeBaseDQuad ):

    def setInitSol( self ):

        self.yCurr = self.bcVec.copy()

        self.setSolVel( self.yCurr, 0 )

    def setRes( self ):
        
        nDims      = self.nDims
        GammaCoefs = self.GammaCoefs
        timeInc    = self.timeInc
        yCurr      = self.yCurr
        yPrev      = self.yPrev
        xFunc      = lambda s : 1.0 if s == nDims else yCurr[s]

        self.res.fill ( 0.0 )

        for m in range( nDims ):
            self.res[m] = yCurr[m] - yPrev[m] -\
                timeInc * yCurr[m + nDims]
            self.res[m + nDims] = yCurr[m + nDims] - yPrev[m + nDims]
            for a in range( nDims ):
                for b in range( nDims ):
                    for s in range( nDims + 1 ):
                        for h in range( nDims + 1 ):
                            self.res[m + nDims] += timeInc * GammaCoefs[m][a][b][s][h] *\
                                xFunc(s) * xFunc(h) *\
                                yCurr[a + nDims] * yCurr[b + nDims] 

        return

    def setLhs( self ):
        
        nDims      = self.nDims
        GammaCoefs = self.GammaCoefs
        timeInc    = self.timeInc
        yCurr      = self.yCurr
        delta      = lambda a,b: 1 if a == b else 0
        xFunc      = lambda s : 1.0 if s == nDims else yCurr[s]

        self.lhs.fill( 0.0 )

        for m in range( nDims ):
            for l in range( nDims ):

                self.lhs[m][l]                 = delta(m,l)
                self.lhs[m][l + nDims]         = -timeInc * delta(m,l)
                self.lhs[m + nDims][l + nDims] = delta(m,l)

                for a in range( nDims ):
                    for b in range( nDims ):
                        for s in range( nDims + 1 ):
                            self.lhs[m + nDims][l] += 2.0 * timeInc *\
                                GammaCoefs[m][a][b][s][l] *\
                                xFunc(s) * yCurr[a + nDims] * yCurr[b + nDims]

                for a in range( nDims ):
                    for s in range( nDims + 1 ):
                        for h in range( nDims + 1 ):
                            self.lhs[m + nDims][l + nDims] += 2.0 * timeInc *\
                                GammaCoefs[m][a][l][s][h] *\
                                xFunc(s) * xFunc(h) * yCurr[a + nDims] 

        return

    def solve( self ):

        nSteps  = self.nSteps
        verbose = self.verbose
        
        self.setInitSol()

        if self.verbose > 0:
            print( '\nSolving geodesic equation...\n' )

        for stepId in range( nSteps ):
            
            if verbose > 1:
                print( 'Solving step', stepId + 1 )
                
            sFlag = self.solveStep()
            
            if not sFlag:
                break

            if verbose > 1:
                print( 'Current solution :', self.yCurr )

            self.setSolVel( self.yCurr, stepId + 1 )

        return sFlag 
    
    def solveStep( self ):

        self.yPrev = self.yCurr.copy()

        convFlag   = False
        
        for itr in range( self.nMaxItrs ):

            self.setRes()
            self.setLhs()

            res     = self.res
            lhs     = self.lhs
            
            delSol  = np.linalg.solve( lhs, -res )

            self.yCurr = self.yCurr + delSol

            errRes = np.linalg.norm( res ) 
            errDel = np.linalg.norm( delSol ) 
            err    = max( errRes, errDel )

            if self.verbose > 1:
                print( 'err =', err )

            if  err < self.tol:
                convFlag = True
                break
            
        return convFlag

# ***********************************************************************
# OdeAdjDQuad: Adjoint ODE solver; 2nd order; for discrete adjoint
# ***********************************************************************

class OdeAdjDQuad( OdeBaseDQuad ):

    def setRhs( self, tsId ):

        nDims   = self.nDims
        nTimes  = self.nTimes
        actSol  = self.actSol
        actVel  = self.actVel
        adjSol  = self.adjSol
        adjVel  = self.adjVel
        yCurr   = self.yCurr
        yPrev   = self.yPrev
        wts     = self.varWeights

        assert len( yPrev ) == 2 * nDims, 'Incorrect array size!'
        assert tsId < nTimes, 'Incorrect tsId!'

        self.res.fill ( 0.0 )

        for l in range( nDims ):

            self.res[l] = yPrev[l] +\
                self.solFuncFct *\
                wts[l] * ( actSol[l][tsId] - adjSol[l][tsId] )
            self.res[l + nDims] = yPrev[l + nDims] +\
                self.velFuncFct *\
                wts[l] * ( actVel[l][tsId] - adjVel[l][tsId] )

    def setLhs( self, tsId ):

        nDims      = self.nDims
        nTimes     = self.nTimes
        timeInc    = self.timeInc
        GammaCoefs = self.GammaCoefs
        adjSol     = self.adjSol
        adjVel     = self.adjVel
        yCurr      = self.yCurr
        yPrev      = self.yPrev
        delta      = lambda a,b: 1 if a == b else 0
        xFunc      = lambda s : 1.0 if s == nDims else adjSol[s][tsId]
        
        assert tsId < nTimes, 'Incorrect tsId!'

        self.lhs.fill (  0.0 )

        for l in range( nDims ):
            for q in range( nDims ):

                self.lhs[l][q]                 = delta(l,q)
                self.lhs[l + nDims][q]         = -timeInc * delta(l,q)

                for a in range( nDims ):
                    for b in range( nDims ):
                        for s in range( nDims + 1 ):
                            self.lhs[l][q + nDims] += 2.0 * timeInc *\
                                GammaCoefs[q][a][b][s][l] *\
                                xFunc(s) * adjVel[a][tsId] * adjVel[b][tsId]

                self.lhs[l + nDims][q + nDims] = delta(l,q)
                for a in range( nDims ):
                    for s in range( nDims + 1 ):
                        for h in range( nDims + 1 ):
                            self.lhs[l + nDims][q + nDims] += 2.0 * timeInc *\
                                GammaCoefs[q][a][l][s][h] *\
                                xFunc(s) * xFunc(h) * adjVel[a][tsId] 

    def solve( self ):

        nSteps   = self.nSteps

        if self.verbose > 0:
            print( '\nSolving adjoint geodesic equation...\n' )

        for stepId in range( nSteps-1, -1, -1 ):
            
            self.yPrev = self.yCurr.copy()

            self.setRhs( stepId + 1 )
            self.setLhs( stepId + 1 )
            
            try:
                self.yCurr = np.linalg.solve( self.lhs, self.res )
            except:
                return False
        
            self.setSolVel( self.yCurr, stepId + 1 )

        return True
