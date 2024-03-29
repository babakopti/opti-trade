# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import numpy as np
import scipy as sp

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
                  intgType = 'LSODA',
                  actSol   = None,
                  adjSol   = None,
                  tol      = 1.0e-4,
                  nMaxItrs = 20,
                  varCoefs = None,
                  srcCoefs = None,
                  srcTerm  = None,
                  atnCoefs = None,                  
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

        if varCoefs is not None:
            assert len( varCoefs ) == nDims, 'Incorrect varCoefs size!'
        else:
            varCoefs = np.ones( shape = ( nDims ), dtype = 'd' )
            
        if srcTerm is not None:
            assert srcTerm.shape[0] == nDims,  'Incorrect srcTerm size!'
            assert srcTerm.shape[1] == nTimes, 'Incorrect srcTerm size!'

        if srcCoefs is not None:
            assert srcCoefs.shape[0] == nDims,     'Incorrect srcCoefs size!'
            assert srcCoefs.shape[1] == nDims + 1, 'Incorrect srcCoefs size!'
        else:
            srcCoefs = np.zeros( shape = ( nDims, nDims + 1 ), dtype = 'd' )

        if atnCoefs is not None:
            assert len( atnCoefs ) == nTimes, 'Incorrect atnCoefs size!'
        else:
            atnCoefs = np.ones( shape = ( nTimes ), dtype = 'd' )
            
        self.Gamma    = Gamma
        self.bcVec    = bcVec
        self.bcTime   = bcTime
        self.nDims    = nDims
        self.timeInc  = timeInc
        self.nSteps   = nSteps
        self.intgType = intgType
        self.actSol   = actSol
        self.adjSol   = adjSol
        self.varCoefs = varCoefs
        self.srcCoefs = srcCoefs
        self.srcTerm  = srcTerm
        self.atnCoefs = atnCoefs                
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
        
        nDims   = self.nDims
        nSteps  = self.nSteps
        nTimes  = self.nTimes
        bcTime  = self.bcTime
        timeInc = self.timeInc

        bkFlag  = bool( bcTime > 0 )

        if bkFlag:
            endTime  = bcTime - nSteps * timeInc
        else:
            endTime  = bcTime + nSteps * timeInc

        timeSpan = ( bcTime, endTime )        
        timeEval = np.linspace( bcTime, endTime, nTimes )

        if self.intgType in [ 'Radau', 'BDF', 'LSODA' ]:
            res = sp.integrate.solve_ivp( fun      = self.fun, 
                                          jac      = self.jac,
                                          y0       = self.bcVec, 
                                          t_span   = timeSpan,
                                          t_eval   = timeEval,
                                          method   = self.intgType,
                                          rtol     = self.tol            )
        else:
            res = sp.integrate.solve_ivp( fun      = self.fun, 
                                          y0       = self.bcVec, 
                                          t_span   = timeSpan,
                                          t_eval   = timeEval,
                                          method   = self.intgType, 
                                          rtol     = self.tol            )
            
        sFlag = res.success
        
        if not sFlag:
            print(res)
            print(self.Gamma)

        assert sFlag, 'Failed to solve the ODE: %s' % res.message
        
        assert res.y.shape[0] == nDims,  'Internal error!'
        assert res.y.shape[1] == nTimes, 'Internal error!'

        if bkFlag:
            self.sol = np.flip( res.y, 1 )
        else:
            self.sol = res.y.copy()

        return sFlag 

    def getSol( self ):
        return self.sol

# ***********************************************************************
# OdeBaseNN(): An Scipy based solver for NN-geo model
# ***********************************************************************

class OdeBaseNN:

    def __init__( self,
                  Gamma,
                  bcVec,
                  bcTime,
                  timeInc,
                  nSteps,
                  intgType = 'LSODA',
                  actSol   = None,
                  adjSol   = None,
                  tol      = 1.0e-4,
                  nMaxItrs = 20,
                  varCoefs = None,
                  srcCoefs = None,
                  srcTerm  = None,
                  atnCoefs = None,                  
                  verbose  = 1           ):

        nDims   = len( bcVec )
        nTimes  = nSteps + 1

        assert bcTime >= 0, 'BC time should be >= 0!'
        
        if actSol is not None:
            assert actSol.shape[0] == nDims,  'Incorrect actSol size!'
            assert actSol.shape[1] == nTimes, 'Incorrect actSol size!'

        if adjSol is not None:
            assert adjSol.shape[0] == nDims,  'Incorrect adjSol size!'
            assert adjSol.shape[1] == nTimes, 'Incorrect adjSol size!'

        if varCoefs is not None:
            assert len( varCoefs ) == nDims, 'Incorrect varCoefs size!'
        else:
            varCoefs = np.ones( shape = ( nDims ), dtype = 'd' )
            
        if srcTerm is not None:
            assert srcTerm.shape[0] == nDims,  'Incorrect srcTerm size!'
            assert srcTerm.shape[1] == nTimes, 'Incorrect srcTerm size!'

        if srcCoefs is not None:
            assert srcCoefs.shape[0] == nDims,     'Incorrect srcCoefs size!'
            assert srcCoefs.shape[1] == nDims + 1, 'Incorrect srcCoefs size!'
        else:
            srcCoefs = np.zeros( shape = ( nDims, nDims + 1 ), dtype = 'd' )

        if atnCoefs is not None:
            assert len( atnCoefs ) == nTimes, 'Incorrect atnCoefs size!'
        else:
            atnCoefs = np.ones( shape = ( nTimes ), dtype = 'd' )
            
        self.Gamma    = Gamma
        self.bcVec    = bcVec
        self.bcTime   = bcTime
        self.nDims    = nDims
        self.timeInc  = timeInc
        self.nSteps   = nSteps
        self.intgType = intgType
        self.actSol   = actSol
        self.adjSol   = adjSol
        self.varCoefs = varCoefs
        self.srcCoefs = srcCoefs
        self.srcTerm  = srcTerm
        self.atnCoefs = atnCoefs                
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
        
        nDims   = self.nDims
        nSteps  = self.nSteps
        nTimes  = self.nTimes
        bcTime  = self.bcTime
        timeInc = self.timeInc

        bkFlag  = bool( bcTime > 0 )

        if bkFlag:
            endTime  = bcTime - nSteps * timeInc
        else:
            endTime  = bcTime + nSteps * timeInc

        timeSpan = ( bcTime, endTime )        
        timeEval = np.linspace( bcTime, endTime, nTimes )

        if self.intgType in [ 'Radau', 'BDF', 'LSODA' ]:
            res = sp.integrate.solve_ivp( fun      = self.fun, 
                                          jac      = self.jac,
                                          y0       = self.bcVec, 
                                          t_span   = timeSpan,
                                          t_eval   = timeEval,
                                          method   = self.intgType, 
                                          rtol     = self.tol            )
        else:
            res = sp.integrate.solve_ivp( fun      = self.fun, 
                                          y0       = self.bcVec, 
                                          t_span   = timeSpan,
                                          t_eval   = timeEval,
                                          method   = self.intgType, 
                                          rtol     = self.tol            )
            
        sFlag = res.success

        assert sFlag, 'Failed to solve the ODE!'
        
        assert res.y.shape[0] == nDims,  'Internal error!'
        assert res.y.shape[1] == nTimes, 'Internal error!'

        if bkFlag:
            self.sol = np.flip( res.y, 1 )
        else:
            self.sol = res.y.copy()

        return sFlag 

    def getSol( self ):
        return self.sol
