# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import numpy as np
import scipy as sp

# ***********************************************************************
# OdeBaseExp(): An Scipy based solver; exp. metric
# ***********************************************************************

class OdeBaseExp:

    def __init__( self,
                  GammaCoefs,
                  bcVec,
                  bcTime,
                  timeInc,
                  nSteps,
                  intgType = 'vode',
                  actSol   = None,
                  actVel   = None,
                  adjObj   = None,
                  tol      = 1.0e-4,
                  nMaxItrs = 20,
                  verbose  = 1           ):

        assert len( bcVec ) % 2 == 0, 'Incorrect bcVec length!'

        nDims   = int( len( bcVec ) / 2 )
        nTimes  = nSteps + 1

        assert bcTime >= 0, 'BC time should be >= 0!'

        assert GammaCoefs.shape[0] == nDims, 'Incorrect Gamma size!'
        assert GammaCoefs.shape[1] == nDims, 'Incorrect Gamma size!'
        
        if actSol is not None:
            assert actSol.shape[0] == nDims,  'Incorrect actSol size!'
            assert actSol.shape[1] == nTimes, 'Incorrect actSol size!'

        if actVel is not None:
            assert actVel.shape[0] == nDims,  'Incorrect actVel size!'
            assert actVel.shape[1] == nTimes, 'Incorrect actVel size!'

        self.GammaCoefs = GammaCoefs
        self.bcVec      = bcVec
        self.bcTime     = bcTime
        self.nDims      = nDims
        self.timeInc    = timeInc
        self.nSteps     = nSteps
        self.intgType   = intgType
        self.actSol     = actSol
        self.actVel     = actVel
        self.adjObj     = adjObj
        self.tol        = tol
        self.nMaxItrs   = nMaxItrs
        self.verbose    = verbose
        self.nTimes     = nTimes

        self.sol        = np.zeros( shape = ( nDims, nTimes ), dtype = 'd' )
        self.vel        = np.zeros( shape = ( nDims, nTimes ), dtype = 'd' )
        self.acl        = np.zeros( shape = ( nDims, nTimes ), dtype = 'd' )
        
    def fun( self, t, y ):
        pass

    def jac( self, t, y ):
        pass

    def solve( self ):
        
        nDims  = self.nDims
        nSteps = self.nSteps

        bkFlag = bool( self.bcTime > 0 )

        odeObj = sp.integrate.ode( self.fun, self.jac )
        odeObj.set_integrator( self.intgType, rtol = self.tol, nsteps = self.nMaxItrs )
        odeObj.set_initial_value( self.bcVec, t = self.bcTime )

        if bkFlag:
            initStepId = nSteps
            fct        = -1.0
        else:
            initStepId = 0
            fct        = 1.0

        tmpVec = self.fun( odeObj.t, odeObj.y )

        for varId in range( nDims ):
            self.sol[varId][initStepId] = odeObj.y[varId]
            self.vel[varId][initStepId] = odeObj.y[varId + nDims]
            self.acl[varId][initStepId] = tmpVec[varId + nDims]

        if self.verbose > 1:
            print( 'Time =', odeObj.t, '; Sol/Vel =', odeObj.y )

        sFlag = True
        for i in range( nSteps-1, -1, -1 ):

            if bkFlag:
                stepId = i
            else:
                stepId = nSteps - i

            odeObj.integrate( odeObj.t + fct * self.timeInc )

            tmpVec = self.fun( odeObj.t, odeObj.y )
            for varId in range( nDims ):
                self.sol[varId][stepId] = odeObj.y[varId]
                self.vel[varId][stepId] = odeObj.y[varId + nDims]
                self.acl[varId][stepId] = tmpVec[varId + nDims]

            sFlag = sFlag & odeObj.successful()

            if not odeObj.successful():
                break

            if self.verbose > 2:
                print( 'Time =', odeObj.t, '; Sol =', odeObj.y )
                print( 'Success:',  odeObj.successful() )

        if self.verbose > 1:
            print( 'Solution success:', sFlag )

        return sFlag 

    def getSol( self ):
        return self.sol

    def getVel( self ):
        return self.vel

    def getAcl( self ):
        return self.acl

# ***********************************************************************
# OdeBaseConst(): An Scipy based solver; const. curv.
# ***********************************************************************

class OdeBaseConst:

    def __init__( self,
                  Gamma,
                  bcVec,
                  bcTime,
                  timeInc,
                  nSteps,
                  intgType = 'vode',
                  actSol   = None,
                  adjSol   = None,
                  tol      = 1.0e-4,
                  nMaxItrs = 20,
                  srcCoefs = None,
                  srcTerm  = None,
                  verbose  = 1           ):

        nDims   = len( bcVec )
        nTimes  = nSteps + 1

        assert bcTime >= 0, 'BC time should be >= 0!'

        assert Gamma.shape[0] == nDims,  'Incorrect Gamma size!'
        assert Gamma.shape[1] == nDims,  'Incorrect Gamma size!'
        assert Gamma.shape[2] == nDims,  'Incorrect Gamma size!'
        
        if actSol is not None:
            assert actSol.shape[0] == nDims,  'Incorrect actSol size!'
            assert actSol.shape[1] == nTimes, 'Incorrect actSol size!'

        if adjSol is not None:
            assert adjSol.shape[0] == nDims,  'Incorrect adjSol size!'
            assert adjSol.shape[1] == nTimes, 'Incorrect adjSol size!'

        if srcTerm is not None:
            assert srcTerm.shape[0] == nDims,  'Incorrect srcTerm size!'
            assert srcTerm.shape[1] == nTimes, 'Incorrect srcTerm size!'

        if srcCoefs is not None:
            assert srcCoefs.shape[0] == nDims,     'Incorrect srcCoefs size!'
            assert srcCoefs.shape[1] == nDims + 1, 'Incorrect srcCoefs size!'
        else:
            srcCoefs = np.zeros( shape = ( nDims, nDims + 1 ), dtype = 'd' )

        self.Gamma    = Gamma
        self.bcVec    = bcVec
        self.bcTime   = bcTime
        self.nDims    = nDims
        self.timeInc  = timeInc
        self.nSteps   = nSteps
        self.intgType = intgType
        self.actSol   = actSol
        self.adjSol   = adjSol
        self.srcCoefs = srcCoefs
        self.srcTerm  = srcTerm
        self.tol      = tol
        self.nMaxItrs = nMaxItrs
        self.verbose  = verbose
        self.nTimes   = nTimes

        self.sol      = np.zeros( shape = ( nDims, nTimes ), dtype = 'd' )

    def fun( self, t, y ):
        pass

    def jac( self, t, y ):
        pass

    def solve( self ):
        
        nDims  = self.nDims
        nSteps = self.nSteps

        bkFlag = bool( self.bcTime > 0 )

        odeObj = sp.integrate.ode( self.fun, self.jac )
        odeObj.set_integrator( self.intgType, rtol = self.tol, nsteps = self.nMaxItrs )
        odeObj.set_initial_value( self.bcVec, t = self.bcTime )

        if self.verbose > 2:
            print( 'Time =', odeObj.t, '; Sol =', odeObj.y )

        if bkFlag:
            initStepId = nSteps
            fct        = -1.0
        else:
            initStepId = 0
            fct        = 1.0

        for varId in range( nDims ):
            self.sol[varId][initStepId] = odeObj.y[varId]

        sFlag = True
        for i in range( nSteps-1, -1, -1 ):

            odeObj.integrate( odeObj.t + fct * self.timeInc )

            if bkFlag:
                stepId = i
            else:
                stepId = nSteps - i

            for varId in range( nDims ):
                self.sol[varId][stepId] = odeObj.y[varId]

            sFlag = sFlag & odeObj.successful()

            if not odeObj.successful():
                break

            if self.verbose > 2:
                print( 'Time =', odeObj.t, '; Sol =', odeObj.y )
                print( 'Success:',  odeObj.successful() )

        if self.verbose > 1:
            print( 'Solution success:', sFlag )

        return sFlag 

    def getSol( self ):
        return self.sol

# ***********************************************************************
# OdeBase(): An Scipy based solvee; 1st order geodesic equation
# ***********************************************************************

class OdeBase:

    def __init__( self,
                  Gamma,
                  bcVec,
                  bcTime,
                  timeInc,
                  nSteps,
                  intgType = 'vode',
                  actSol   = None,
                  adjSol   = None,
                  tol      = 1.0e-4,
                  nMaxItrs = 20,
                  verbose  = 1           ):

        nDims   = len( bcVec )
        nTimes  = nSteps + 1

        assert bcTime >= 0, 'BC time should be >= 0!'

        assert Gamma.shape[0] == nDims,  'Incorrect Gamma size!'
        assert Gamma.shape[1] == nDims,  'Incorrect Gamma size!'
        assert Gamma.shape[2] == nDims,  'Incorrect Gamma size!'
        assert Gamma.shape[3] == nTimes, 'Incorrect Gamma size!'
        
        if actSol is not None:
            assert actSol.shape[0] == nDims,  'Incorrect actSol size!'
            assert actSol.shape[1] == nTimes, 'Incorrect actSol size!'

        if adjSol is not None:
            assert adjSol.shape[0] == nDims,  'Incorrect adjSol size!'
            assert adjSol.shape[1] == nTimes, 'Incorrect adjSol size!'
                    
        self.Gamma    = Gamma
        self.bcVec    = bcVec
        self.bcTime   = bcTime
        self.nDims    = nDims
        self.timeInc  = timeInc
        self.nSteps   = nSteps
        self.intgType = intgType
        self.actSol   = actSol
        self.adjSol   = adjSol
        self.tol      = tol
        self.nMaxItrs = nMaxItrs
        self.verbose  = verbose
        self.nTimes   = nTimes
        self.funCnt   = 0
        self.jacCnt   = 0

        self.sol      = np.zeros( shape = ( nDims, nTimes ), dtype = 'd' )

    def fun( self, t, y ):
        pass

    def jac( self, t, y ):
        pass

    def solve( self ):
        
        nDims  = self.nDims
        nSteps = self.nSteps

        adjFlag = bool( self.bcTime > 0 )

        if adjFlag:
            if self.verbose > 0:
                print( '\nSolving adjoint...\n' )
        else:
            if self.verbose > 0:
                print( '\nSolving geodesic...\n' )

        odeObj = sp.integrate.ode( self.fun, self.jac )
        odeObj.set_integrator( self.intgType, rtol = self.tol, nsteps = self.nMaxItrs )
        odeObj.set_initial_value( self.bcVec, t = self.bcTime )

        if adjFlag:
            tsId = nSteps 
        else:
            tsId = 0

        for varId in range( nDims ):
            self.sol[varId][tsId] = odeObj.y[varId]

        if self.verbose > 1:
            print( 'Time =', odeObj.t, '; Sol =', odeObj.y )

        sFlag = True
        for stepId in range( nSteps ):

            if adjFlag:
                fct = -1.0
                odeObj.set_f_params(   nSteps - 1 - stepId )
                odeObj.set_jac_params( nSteps - 1 - stepId )
            else:
                fct = 1.0
                odeObj.set_f_params(   stepId )
                odeObj.set_jac_params( stepId )

            odeObj.integrate( odeObj.t + fct * self.timeInc )

            if adjFlag:
                tsId = nSteps - 1 - stepId 
            else:
                tsId = stepId + 1

            for varId in range( nDims ):
                self.sol[varId][tsId] = odeObj.y[varId]

            sFlag = sFlag & odeObj.successful()

            if not odeObj.successful():
                break

            if self.verbose > 1:
                print( 'Time =', odeObj.t, '; Sol =', odeObj.y )
                print( 'Success:',  odeObj.successful() )

        if self.verbose > 0:
            print( 'Solution success:', sFlag )

        return sFlag 

    def getSol( self ):
        return self.sol

# ***********************************************************************
# OdeBaseLin(): An Scipy based solver; 2nd order geodesic, linear curv.
# ***********************************************************************

class OdeBaseLin:

    def __init__( self,
                  GammaCoefs,
                  bcVec,
                  bcTime,
                  timeInc,
                  nSteps,
                  intgType = 'vode',
                  actSol   = None,
                  actVel   = None,
                  actAcl   = None,
                  adjObj   = None,
                  tol      = 1.0e-4,
                  nMaxItrs = 20,
                  verbose  = 1           ):

        assert len( bcVec ) % 2 == 0, 'Incorrect bcVec length!'

        nDims   = int( len( bcVec ) / 2 )
        nTimes  = nSteps + 1
        nTmp    = nDims  + 1

        assert bcTime >= 0, 'BC time should be >= 0!'

        assert GammaCoefs.shape[0] == nDims, 'Incorrect Gamma size!'
        assert GammaCoefs.shape[1] == nDims, 'Incorrect Gamma size!'
        assert GammaCoefs.shape[2] == nDims, 'Incorrect Gamma size!'
        assert GammaCoefs.shape[3] == nTmp,  'Incorrect Gamma size!'
        
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
        self.bcVec      = bcVec
        self.bcTime     = bcTime
        self.nDims      = nDims
        self.timeInc    = timeInc
        self.nSteps     = nSteps
        self.intgType   = intgType
        self.actSol     = actSol
        self.actVel     = actVel
        self.actAcl     = actAcl
        self.adjObj     = adjObj
        self.tol        = tol
        self.nMaxItrs   = nMaxItrs
        self.verbose    = verbose
        self.nTimes     = nTimes

        self.sol        = np.zeros( shape = ( nDims, nTimes ), dtype = 'd' )
        self.vel        = np.zeros( shape = ( nDims, nTimes ), dtype = 'd' )
        self.acl        = np.zeros( shape = ( nDims, nTimes ), dtype = 'd' )
        
    def fun( self, t, y ):
        pass

    def jac( self, t, y ):
        pass

    def solve( self ):
        
        nDims  = self.nDims
        nSteps = self.nSteps

        adjFlag = bool( self.bcTime > 0 )

        if adjFlag:
            if self.verbose > 0:
                print( '\nSolving adjoint...\n' )
        else:
            if self.verbose > 0:
                print( '\nSolving geodesic...\n' )

        odeObj = sp.integrate.ode( self.fun, self.jac )
        odeObj.set_integrator( self.intgType, rtol = self.tol, nsteps = self.nMaxItrs )
        odeObj.set_initial_value( self.bcVec, t = self.bcTime )

        if adjFlag:
            tsId = nSteps 
        else:
            tsId = 0

        tmpVec = self.fun( odeObj.t, odeObj.y, -1 )
        for i in range( nDims ):
            self.sol[i][tsId] = odeObj.y[ i ]
            self.vel[i][tsId] = odeObj.y[ i + nDims ]
            self.acl[i][tsId] = tmpVec[ i + nDims ]

        if self.verbose > 1:
            print( 'Time =', odeObj.t, '; Sol/Vel =', odeObj.y )

        sFlag = True
        for stepId in range( nSteps ):

            if adjFlag:
                fct = -1.0
                odeObj.set_f_params(   nSteps - 1 - stepId )
                odeObj.set_jac_params( nSteps - 1 - stepId )
            else:
                fct = 1.0
                odeObj.set_f_params(   stepId )
                odeObj.set_jac_params( stepId )

            odeObj.integrate( odeObj.t + fct * self.timeInc )

            if adjFlag:
                tsId = nSteps - 1 - stepId 
            else:
                tsId = stepId + 1

            tmpVec = self.fun( odeObj.t, odeObj.y, stepId )
            for i in range( nDims ):
                self.sol[i][tsId] = odeObj.y[ i ]
                self.vel[i][tsId] = odeObj.y[ i + nDims ]
                self.acl[i][tsId] = tmpVec[ i + nDims ]

            sFlag = sFlag & odeObj.successful()

            if not odeObj.successful():
                break

            if self.verbose > 1:
                print( 'Time =', odeObj.t, '; Sol/Vel =', odeObj.y )
                print( 'Success:',  odeObj.successful() )

        if self.verbose > 0:
            print( 'Solution success:', sFlag )

        return sFlag 

    def getSol( self ):
        return self.sol

    def getVel( self ):
        return self.vel

    def getAcl( self ):
        return self.acl

# ***********************************************************************
# OdeBaseDLin: Basic ODE solver; 2nd order; for discrete adjoint
# ***********************************************************************

class OdeBaseDLin():

    def __init__( self,
                  GammaCoefs,
                  nDims,
                  nSteps,
                  timeInc,
                  bcVec      = None,
                  actSol     = None,
                  adjSol     = None,
                  actVel     = None,
                  adjVel     = None,
                  varWeights = None,
                  solFuncFct = 0, 
                  velFuncFct = 1, 
                  tol        = 1.0e-4,
                  nMaxItrs   = 20,
                  verbose    = 1           ):

        nTimes  = nSteps + 1
        nTmp    = nDims  + 1

        assert GammaCoefs.shape[0] == nDims, 'Incorrect Gamma size!'
        assert GammaCoefs.shape[1] == nDims, 'Incorrect Gamma size!'
        assert GammaCoefs.shape[2] == nDims, 'Incorrect Gamma size!'
        assert GammaCoefs.shape[3] == nTmp,  'Incorrect Gamma size!'

        if bcVec is not None:
            assert len( bcVec ) == 2 * nDims, 'Incorrect bcVec size!'

        if actSol is not None:
            assert actSol.shape[0] == nDims,  'Incorrect actSol size!'
            assert actSol.shape[1] == nTimes, 'Incorrect actSol size!'

        if actVel is not None:
            assert actVel.shape[0] == nDims,  'Incorrect actVel size!'
            assert actVel.shape[1] == nTimes, 'Incorrect actVel size!'

        if adjSol is not None:
            assert adjSol.shape[0] == nDims,  'Incorrect adjSol size!'
            assert adjSol.shape[1] == nTimes, 'Incorrect adjSol size!'

        if adjVel is not None:
            assert adjVel.shape[0] == nDims,  'Incorrect adjVel size!'
            assert adjVel.shape[1] == nTimes, 'Incorrect adjVel size!'

        if varWeights is not None:
            assert len( varWeights ) == nDims,  'Incorrect adjVel size!'

        self.GammaCoefs = GammaCoefs
        self.bcVec      = bcVec
        self.actSol     = actSol
        self.adjSol     = adjSol
        self.actVel     = actVel
        self.adjVel     = adjVel
        self.varWeights = varWeights
        self.nDims      = nDims
        self.timeInc    = timeInc
        self.nSteps     = nSteps
        self.nTimes     = nTimes
        self.solFuncFct = solFuncFct
        self.velFuncFct = velFuncFct
        self.tol        = tol
        self.nMaxItrs   = nMaxItrs
        self.verbose    = verbose

        self.sol      = np.zeros( shape = ( nDims, nTimes ),  dtype = 'd' )
        self.vel      = np.zeros( shape = ( nDims, nTimes ),  dtype = 'd' )
        self.yCurr    = np.zeros( shape = ( 2 * nDims ),      dtype = 'd' )
        self.yPrev    = np.zeros( shape = ( 2 * nDims ),      dtype = 'd' )
        self.res      = np.zeros( shape = ( 2 * nDims ),      dtype = 'd' )
        self.lhs      = np.zeros( shape = ( 2 * nDims, 2 * nDims ), 
                                  dtype = 'd' )

    def setSolVel( self, y, tsId ):

        nDims  = self.nDims
        nTimes = self.nTimes

        assert len( y ) == 2 * nDims, 'Incorrect array size!'
        assert tsId < nTimes, 'Incorrect tsId!'

        for m in range( nDims ):
            self.sol[m][tsId] = y[m]
            self.vel[m][tsId] = y[m + nDims]

    def getSol( self ):

        return self.sol

    def getVel( self ):

        return self.vel

# ***********************************************************************
# OdeBaseDQuad: Basic ODE solver; 2nd order; for discrete adjoint
# ***********************************************************************

class OdeBaseDQuad():

    def __init__( self,
                  GammaCoefs,
                  nDims,
                  nSteps,
                  timeInc,
                  bcVec      = None,
                  actSol     = None,
                  adjSol     = None,
                  actVel     = None,
                  adjVel     = None,
                  varWeights = None,
                  solFuncFct = 0, 
                  velFuncFct = 1, 
                  tol        = 1.0e-4,
                  nMaxItrs   = 20,
                  verbose    = 1           ):

        nTimes  = nSteps + 1
        nTmp    = nDims + 1

        assert GammaCoefs.shape[0] == nDims, 'Incorrect Gamma size!'
        assert GammaCoefs.shape[1] == nDims, 'Incorrect Gamma size!'
        assert GammaCoefs.shape[2] == nDims, 'Incorrect Gamma size!'
        assert GammaCoefs.shape[3] == nTmp,  'Incorrect Gamma size!'
        assert GammaCoefs.shape[4] == nTmp,  'Incorrect Gamma size!'

        if bcVec is not None:
            assert len( bcVec ) == 2 * nDims, 'Incorrect bcVec size!'

        if actSol is not None:
            assert actSol.shape[0] == nDims,  'Incorrect actSol size!'
            assert actSol.shape[1] == nTimes, 'Incorrect actSol size!'

        if actVel is not None:
            assert actVel.shape[0] == nDims,  'Incorrect actVel size!'
            assert actVel.shape[1] == nTimes, 'Incorrect actVel size!'

        if adjSol is not None:
            assert adjSol.shape[0] == nDims,  'Incorrect adjSol size!'
            assert adjSol.shape[1] == nTimes, 'Incorrect adjSol size!'

        if adjVel is not None:
            assert adjVel.shape[0] == nDims,  'Incorrect adjVel size!'
            assert adjVel.shape[1] == nTimes, 'Incorrect adjVel size!'

        if varWeights is not None:
            assert len( varWeights ) == nDims,  'Incorrect adjVel size!'

        self.GammaCoefs = GammaCoefs
        self.bcVec      = bcVec
        self.actSol     = actSol
        self.adjSol     = adjSol
        self.actVel     = actVel
        self.adjVel     = adjVel
        self.varWeights = varWeights
        self.nDims      = nDims
        self.timeInc    = timeInc
        self.nSteps     = nSteps
        self.nTimes     = nTimes
        self.solFuncFct = solFuncFct
        self.velFuncFct = velFuncFct
        self.tol        = tol
        self.nMaxItrs   = nMaxItrs
        self.verbose    = verbose

        self.sol      = np.zeros( shape = ( nDims, nTimes ),  dtype = 'd' )
        self.vel      = np.zeros( shape = ( nDims, nTimes ),  dtype = 'd' )
        self.yCurr    = np.zeros( shape = ( 2 * nDims ),      dtype = 'd' )
        self.yPrev    = np.zeros( shape = ( 2 * nDims ),      dtype = 'd' )
        self.res      = np.zeros( shape = ( 2 * nDims ),      dtype = 'd' )
        self.lhs      = np.zeros( shape = ( 2 * nDims, 2 * nDims ), 
                                  dtype = 'd' )

    def setSolVel( self, y, tsId ):

        nDims  = self.nDims
        nTimes = self.nTimes

        assert len( y ) == 2 * nDims, 'Incorrect array size!'
        assert tsId < nTimes, 'Incorrect tsId!'

        for m in range( nDims ):
            self.sol[m][tsId] = y[m]
            self.vel[m][tsId] = y[m + nDims]

    def getSol( self ):

        return self.sol

    def getVel( self ):

        return self.vel
