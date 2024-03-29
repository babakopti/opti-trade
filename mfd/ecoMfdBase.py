# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import math
import time
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D
import pickle as pk
import dill
import gc

from scipy.integrate import trapz
from scipy.optimize import line_search
from scipy.optimize import fsolve
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

sys.path.append( os.path.abspath( '../' ) )

from utl.utils import getLogger

# ***********************************************************************
# Some parameters
# ***********************************************************************

SET_VAR_OFFSET = False

# ***********************************************************************
# Class EcoMfdCBase: Base economic manifold ; continues adjoint
# ***********************************************************************

class EcoMfdCBase:

    def __init__(   self,
                    varNames,
                    velNames,
                    dateName, 
                    dfFile,
                    minTrnDate, 
                    maxTrnDate,
                    maxOosDate,
                    trmFuncDict  = {},
                    optType      = 'L-BFGS-B',
                    maxOptItrs   = 100, 
                    optGTol      = 1.0e-4,
                    optFTol      = 1.0e-8,
                    stepSize     = None,
                    factor       = 4.0e-5,
                    regCoef      = 0.0,
                    regL1Wt      = 0.0,
                    nPca         = None,
                    endBcFlag    = True,
                    varCoefs     = None,
                    srcCoefs     = None,
                    srcTerm      = None,
                    atnFct       = 1.0,
                    mode         = 'intraday',
                    logFileName  = None,
                    verbose      = 1        ):

        assert len( varNames ) == len( velNames ),\
            'Inconsistent variable and velocity lists!'

        assert pd.to_datetime( maxOosDate ) > pd.to_datetime( maxTrnDate ),\
            'maxOosDate should be larger than maxTrnDate!'

        assert regL1Wt >= 0, 'Incorrect value; regL1Wt should be >= 0!'
        assert regL1Wt <= 1, 'Incorrect value; regL1Wt should be <= 1!'

        self.varNames    = varNames
        self.velNames    = velNames
        self.dateName    = dateName
        self.dfFile      = dfFile
        self.minTrnDate  = minTrnDate
        self.maxTrnDate  = maxTrnDate
        self.maxOosDate  = maxOosDate
        self.trmFuncDict = trmFuncDict 
        self.optType     = optType,
        self.optType     = self.optType[0]
        self.maxOptItrs  = maxOptItrs
        self.optGTol     = optGTol
        self.optFTol     = optFTol
        self.stepSize    = stepSize
        self.factor      = factor
        self.regCoef     = regCoef
        self.regL1Wt     = regL1Wt
        self.endBcFlag   = endBcFlag
        self.mode        = mode
        self.trnDf       = None
        self.oosDf       = None
        self.errVec      = []
        self.nDims       = len( varNames )
        self.statHash    = {}
        self.deNormHash  = {}
        self.logFileName = logFileName
        self.verbose     = verbose
        self.logger      = getLogger( logFileName, verbose, 'mfd' )
        
        self.statHash[ 'funCnt' ]     = 0
        self.statHash[ 'gradCnt' ]    = 0
        self.statHash[ 'odeCnt' ]     = 0
        self.statHash[ 'adjOdeCnt' ]  = 0
        self.statHash[ 'odeTime' ]    = 0.0
        self.statHash[ 'adjOdeTime' ] = 0.0
        self.statHash[ 'totTime' ]    = 0.0

        self.pcaFlag = False
     
        if nPca is not None:

            self.pcaFlag = True

            if nPca == 'full':
                self.nPca = self.nDims
            else:
                self.nPca = nPca

            self.pca = PCA( n_components = self.nPca )

        if varCoefs is None:
            self.varCoefs = np.empty( shape = ( self.nDims ), dtype = 'd' )
            self.varCoefs.fill ( 1.0 )
        else:
            assert len( varCoefs ) == self.nDims, 'Incorrect size for varCoefs!'
            self.varCoefs = varCoefs

        self.srcCoefs    = srcCoefs
        self.srcTerm     = srcTerm

        nDims            = self.nDims
        self.endSol      = np.zeros( shape = ( nDims ), dtype = 'd' )
        self.stdVec      = np.zeros( shape = ( nDims ), dtype = 'd' )        
        self.varOffsets  = np.zeros( shape = ( nDims ), dtype = 'd' )

        self.setDf()        

        self.nSteps      = self.nTimes - 1
        nTimes           = self.nTimes
        nOosTimes        = self.nOosTimes
        self.actSol      = np.zeros( shape = ( nDims, nTimes ),    dtype = 'd' )
        self.actOosSol   = np.zeros( shape = ( nDims, nOosTimes ), dtype = 'd' )
        self.tmpVec      = np.zeros( shape = ( nTimes ), 	   dtype = 'd' )     

        self.trnEndDate  = list( self.trnDf[ dateName ] )[-1]

        self.atnCoefs = np.ones( shape = ( nTimes ) )

        self.setAtnCoefs( atnFct )

    def setDf( self ):

        t0             = time.time()
        dfFile         = self.dfFile
        dateName       = self.dateName
        nDims          = self.nDims
        varNames       = self.varNames
        velNames       = self.velNames
        minDt          = pd.to_datetime( self.minTrnDate )
        maxDt          = pd.to_datetime( self.maxTrnDate )
        maxOosDt       = pd.to_datetime( self.maxOosDate )
        fileExt        = dfFile.split( '.' )[-1]

        if fileExt == 'csv':
            df = pd.read_csv( dfFile ) 
        elif fileExt == 'pkl':
            df = pd.read_pickle( dfFile ) 
        else:
            assert False, 'Unknown input file extension %s' % fileExt

        if self.mode == 'day':
            tmpFunc        = lambda x : pd.to_datetime( x ).date()
            df[ 'tmp' ]    = df[ dateName ].apply( tmpFunc )
            df             = df.groupby( [ 'tmp' ] )[ velNames ].mean()
            df[ dateName ] = df.index
            df             = df.reset_index( drop = True )
        elif self.mode == 'intraday':
            pass
        else:
            assert False, 'Mode %s is not supported!' % self.mode 
        
        df             = df[ [ dateName ] + velNames ]
        df[ dateName ] = df[ dateName ].apply( pd.to_datetime )
        df             = df.interpolate( method = 'linear' )
        df             = df.dropna()
        df             = df[ df[ dateName ] >= minDt ]
        df             = df[ df[ dateName ] <= maxOosDt ]
        df             = self.trmVars( df )
        df             = df.dropna()
        df             = df.sort_values( [ dateName ] )
        df             = df.reset_index( drop = True )
        df[ 'time' ]   = df.index

        if False:
            times        = df[ dateName ] - df[ dateName ][0]
            times        = times.apply( lambda x : x.total_seconds() / 60.0 )
            df[ 'time' ] = times

        if self.pcaFlag:
            df       = self.setPcaVars( df )

        dates      = np.array( df[ dateName ] )
        nRows      = df.shape[0]
        trnCnt     = 0
        for rowId in range( nRows ):
            if dates[rowId] == maxDt:
                trnCnt = rowId + 1
                break
            elif dates[rowId] > maxDt:
                trnCnt = rowId
                break

        oosCnt     = nRows - trnCnt + 1

        self.trnDf = df.head( trnCnt )
        self.oosDf = df.tail( oosCnt )

        self.nTimes      = self.trnDf.shape[0]
        self.nOosTimes   = self.oosDf.shape[0]

        try:
            self.setVarOffsets()
        except:
            pass

        self.logger.info( 'Setting data frame: %0.2f seconds', 
                          time.time() - t0  )

    def trmVars( self, df ):
 
        nDims          = self.nDims
        velNames       = self.velNames
        trmFuncDict    = self.trmFuncDict

        for varId in range( nDims ):

            varVel = velNames[varId]

            self.logger.debug( 'Transforming ' + varVel )

            if varVel in trmFuncDict.keys():
                trmFunc      = trmFuncDict[ varVel ]
                df[ 'TMP' ]  = trmFunc( df[ varVel ] )
                df[ 'TMP' ]  = df[ 'TMP' ].fillna( df[ varVel ] )
                df[ varVel ] = df[ 'TMP' ]

            fct          = self.factor
            velMax       = np.max(  df[ varVel ] )
            velMin       = np.min(  df[ varVel ] )
            df[ varVel ] = ( df[ varVel ] - velMin ) / ( velMax - velMin )
            df[ varVel ] = df[ varVel ] * fct

            self.deNormHash[ varVel ] = ( ( velMax - velMin ) / fct, velMin )

        return df

    def setPcaVars( self, df ):

        velNames = self.velNames

        X        = np.array( df[ velNames ] )

        pcaVec   = self.pca.fit_transform( X )
        
        self.velNames = []
        for varId in range( self.nPca ):
            newVelName       = 'var_pca_' + str( varId )
            df[ newVelName ] = pcaVec[:,varId] 
            self.velNames.append( newVelName )

        self.nDims = self.nPca

        return df

    def setVarOffsets( self ):

        if not SET_VAR_OFFSET:
            return
        
        nDims          = self.nDims
        nTimes         = self.nTimes
        dateName       = self.dateName
        varNames       = self.varNames
        trnDf          = self.trnDf
        dfFile         = self.dfFile
        fileExt        = dfFile.split( '.' )[-1]

        if fileExt == 'csv':
            df = pd.read_csv( dfFile ) 
        elif fileExt == 'pkl':
            df = pd.read_pickle( dfFile ) 
        else:
            assert False, 'Unknown input file extension %s' % fileExt
            
        df             = df[ [dateName] + varNames ]
        df[ dateName ] = pd.to_datetime( df[ dateName ] )
        df             = df.interpolate( method = 'linear' )
        df             = trnDf.merge( df, how = 'left', on = dateName )
        df             = df[ varNames ]

        for m in range( nDims ):
            varName            = varNames[m]
            self.varOffsets[m] = list( df[varName] )[nTimes-1]

    def setAtnCoefs( self, atnFct ):

        assert atnFct >= 0.0, 'atnFct should be positive!'
        assert atnFct <= 1.0, 'atnFct should be less than or equal to 1.0!'

        nTimes = self.nTimes
        tmp    = ( 1.0 - atnFct ) / ( nTimes - 1 )

        for tsId in range( nTimes ):
            self.atnCoefs[tsId] = atnFct + tmp * tsId

        return

    def setParams( self ):

        self.logger.info( 'Running continuous adjoint optimization to set Christoffel symbols...' )

        t0 = time.time()

        if self.optType == 'GD':
            sFlag = self.setParamsGD()
        else:

            options  = { 'gtol'       : self.optGTol, 
                         'ftol'       : self.optFTol, 
                         'maxiter'    : self.maxOptItrs,
                         'eps'        : 0.001,
                         'disp'       : True              }

            tmp_params = np.load(open("/Users/babak/Desktop/params.npy", "rb"))
            print("Babak val:", self.getObjFunc(tmp_params)); #sys.exit()
            try:
                bounds = [(-1.0, 1.0) for i in range(self.nParams)]
                optObj = scipy.optimize.minimize( fun      = self.getObjFunc, 
                                                  x0       = self.params,
                                                  method   = self.optType, 
                                                  jac      = self.getGrad,
                                                  bounds   = bounds,
                                                  options  = options          )
                sFlag   = optObj.success
    
                self.params = optObj.x

                self.logger.info( 'Success: %s', str( sFlag ) )

            except Exception as exc:
                self.logger.error(exc)
                sFlag = False

        self.statHash[ 'totTime' ] = time.time() - t0

        self.logger.info( 'Setting Gamma: %0.2f seconds.', 
                          time.time() - t0   )

        self.setConstStdVec()
        
        return sFlag

    def setParamsGD( self ):

        if self.stepSize is None:
            stepSize = 1.0
            lsFlag   = True
            self.logger.info( 'Line search enabled as step size is not set!' )
        else:
            stepSize = self.stepSize
            lsFlag   = False

        for itr in range( self.maxOptItrs ):
                
            grad     = self.getGrad( self.params )
            funcVal  = self.getObjFunc( self.params )   

            if itr == 0:
                funcVal0  = funcVal
                norm0     = np.linalg.norm( grad )
                
            if lsFlag:
                obj = scipy.optimize.line_search( f        = self.getObjFunc, 
                                                  myfprime = self.getGrad, 
                                                  xk       = self.params,
                                                  pk       = -grad, 
                                                  gfk      = grad,
                                                  old_fval = funcVal,
                                                  c1       = 0.1, 
                                                  c2       = 0.9, 
                                                  maxiter  = 3       )
            
                if obj[0] is not None:
                    stepSize = obj[0]
                else:
                    self.logger.warning( 'Line search did not converge! Using previous value!' )

            tmp = np.linalg.norm( grad ) / norm0

            self.logger.debug( 'Iteration %d: step size     = %.8f', itr + 1, stepSize )

            self.logger.info( 'Iteration %d: rel. gradient norm = %.8f', itr + 1, tmp )
            
            if tmp < self.optGTol:
                self.logger.info( 'Converged at iteration %d; rel. gradient norm = %.8f', itr + 1, tmp )
                return True

            tmp = funcVal / funcVal0

            if tmp < self.optFTol:
                self.logger.info( 'Converged at iteration %d; rel. func. val. = %.8f', itr + 1, tmp )
                return True

            self.params = self.params - stepSize * grad

        gc.collect()
        
        return False

    def setConstStdVec( self ): 

        nDims  = self.nDims
        actSol = self.actSol
        odeObj  = self.getSol( self.params )
        sol     = odeObj.getSol()

        for varId in range( nDims ):
            tmpVec = ( sol[varId][:] - actSol[varId][:] )**2
            self.stdVec[varId] = np.sqrt( np.mean( tmpVec ) )

    def getObjFunc( self, params ):

        self.statHash[ 'funCnt' ] += 1

        nDims    = self.nDims
        nTimes   = self.nTimes
        regCoef  = self.regCoef
        regL1Wt  = self.regL1Wt
        actSol   = self.actSol
        varCoefs = self.varCoefs
        atnCoefs = self.atnCoefs        
        tmpVec   = self.tmpVec.copy()

        odeObj  = self.getSol(params)
        
        if odeObj is None:
            return np.inf

        sol     = odeObj.getSol()

        tmpVec.fill( 0.0 )
        for varId in range( nDims ):
            tmpVec += varCoefs[varId] * atnCoefs[:] *\
                ( sol[varId][:] - actSol[varId][:] )**2 

        val = 0.5 * trapz( tmpVec, dx = 1.0 )

        tmp1  = np.linalg.norm( params, 1 )
        tmp2  = np.linalg.norm( params )
        val  += regCoef * ( regL1Wt * tmp1 + ( 1.0 - regL1Wt ) * tmp2**2 )

        del sol
        del tmpVec
        
        gc.collect()

        return val

    def intgVel( self, y, varId, trnFlag = True ):

        nTimes = len( y )

        if trnFlag:
            assert nTimes == self.nTimes, 'Incorrect size!'

        yCum   = np.zeros( shape = ( nTimes ), dtype = 'd' )

        if trnFlag:
            for tsId in range( nTimes-2, 0, -1 ):
                yCum[tsId] = yCum[tsId+1] - y[tsId]
        else:
            for tsId in range( 1, nTimes ):
                yCum[tsId] = yCum[tsId-1] + y[tsId]

        yCum += self.varOffsets[varId]
                         
        return yCum

    def intgVelStd( self, velStd, nTimes, trnFlag = True ):

        if trnFlag:
            assert nTimes == self.nTimes, 'Incorrect size!'

        varStdVec = np.zeros( shape = ( nTimes ), dtype = 'd' )

        varStdVec.fill( velStd**2 )

        if trnFlag:
            for tsId in range( nTimes-2, 0, -1 ):
                varStdVec[tsId] = varStdVec[tsId+1] + velStd**2
        else:
            for tsId in range( 1, nTimes ):
                varStdVec[tsId] = varStdVec[tsId-1] + velStd**2

        varStdVec = np.sqrt( varStdVec )

        return varStdVec

    def getError( self, varNames = None ): 

        if varNames is None:
            varNames = self.varNames

        nTimes   = self.nTimes
        nDims    = self.nDims
        actSol   = self.actSol
        varCoefs = self.varCoefs
        atnCoefs = self.atnCoefs
        tmpVec   = np.zeros( shape = ( nTimes ), dtype = 'd' )

        tmpVec.fill( 0.0 )
        for varId in range( nDims ):

            varName = self.varNames[varId]

            if varName not in varNames:
                continue

            tmpVec += varCoefs[varId] * atnCoefs[:] * actSol[varId][:]**2 

        funcValFct = 0.5 * trapz( tmpVec, dx = 1.0 )

        if  funcValFct > 0:
            funcValFct = 1.0 / funcValFct

        odeObj  = self.getSol( self.params )
        
        if odeObj is None:
            return -np.inf

        sol      = odeObj.getSol()

        tmpVec.fill( 0.0 )
        for varId in range( nDims ):

            varName = self.varNames[varId]

            if varName not in varNames:
                continue

            tmpVec += varCoefs[varId] * atnCoefs[:] *\
                ( sol[varId][:] - actSol[varId][:] )**2 

        funcVal  = 0.5 * trapz( tmpVec, dx = 1.0 )

        tmpVal   = np.sqrt( funcVal * funcValFct )

        return tmpVal

    def getMerit( self, varNames = None ): 

        tmpVal = self.getError( varNames = varNames ) 
        tmpVal = max( 1.0 - tmpVal, 0.0 )

        return tmpVal

    def getOosError( self, varNames = None ): 

        if varNames is None:
            varNames = self.varNames

        nDims     = self.nDims
        nOosTimes = self.nOosTimes
        actOosSol = self.actOosSol
        varCoefs  = self.varCoefs
        tmpVec    = np.zeros( shape = ( nOosTimes ), dtype = 'd' )

        tmpVec.fill( 0.0 )
        for varId in range( nDims ):

            varName = self.varNames[varId]

            if varName not in varNames:
                continue

            tmpVec += varCoefs[varId] * actOosSol[varId][:]**2 

        funcValFct = 0.5 * trapz( tmpVec, dx = 1.0 )

        if  funcValFct > 0:
            funcValFct = 1.0 / funcValFct

        oosOdeObj  = self.getOosSol()

        if oosOdeObj is None:
            return -np.inf

        oosSol     = oosOdeObj.getSol()

        tmpVec.fill( 0.0 )
        for varId in range( nDims ):

            varName = self.varNames[varId]

            if varName not in varNames:
                continue

            tmpVec += varCoefs[varId] * ( oosSol[varId][:] - actOosSol[varId][:] )**2 

        funcVal  = 0.5 * trapz( tmpVec, dx = 1.0 )

        tmpVal   = np.sqrt( funcVal * funcValFct )

        return tmpVal

    def getOosMerit( self, varNames = None ): 

        tmpVal = self.getOosError( varNames = varNames ) 
        tmpVal = max( 1.0 - tmpVal, 0.0 )

        return tmpVal

    def getOosTrendCnt( self, vType = 'vel' ): 

        nDims = self.nDims
        perfs = self.getOosTrendPerfs( vType )
        cnt   = 0            

        for varId in range( nDims ):
            if perfs[varId]:
                cnt += 1

        assert cnt <= nDims, 'Count cannot be more than nDims!' 

        ratio = float( cnt ) / nDims
               
        return ratio

    def getOosTrendPerfs( self, vType = 'vel' ): 

        nDims     = self.nDims
        nOosTimes = self.nOosTimes
        actOosSol = self.actOosSol

        oosOdeObj  = self.getOosSol()

        if oosOdeObj is None:
            return -np.inf

        oosSol     = oosOdeObj.getSol()

        perfs = []

        for varId in range( nDims ):
            if vType == 'var':
                yAct = self.intgVel( actOosSol[varId], varId, False )
                y    = self.intgVel( oosSol[varId],    varId, False )
            else:
                yAct = actOosSol[varId]
                y    = oosSol[varId]

            actTrend = np.mean( yAct ) - actOosSol[varId][0]
            prdTrend = np.mean( y ) - oosSol[varId][0]
            fct      = actTrend * prdTrend
            
            if fct > 0:
                perfs.append( True )
            else:
                perfs.append( False )
            
        return perfs

    def getRelBias( self, varId ):

        t0 = time.time()
        
        odeObj = self.getSol( self.params )

        if odeObj is None:
            return np.inf

        fct = np.mean( self.actSol[varId] )
        
        if fct != 0:
            fct = 1.0 / fct
            
        val = np.mean( self.actSol[varId] - odeObj.sol[varId] ) * fct

        self.logger.debug( 'Calculating bias took %0.2f seconds', 
                           time.time() - t0  )

        return val

    def getConstStdVec( self ): 

        return self.stdVec

    def getStdCoefs( self ): 

        nDims   = self.nDims
        nTimes  = self.nTimes
        nSteps  = self.nSteps
        times   = np.linspace( 0, nSteps, nTimes )
        bcTime  = times[-1]
        actSol  = self.actSol
        odeObj  = self.getSol( self.params )
        sol     = odeObj.getSol()
        coefs   = np.zeros( shape = ( nDims ), dtype = 'd' )

        def funcVal( alpha, m, sol, actSol, bcTime, times, nTimes ):
            
            tmpVal = 0.0
            for i in range( nTimes ):
                fct     = abs( times[i] - bcTime )
                tanVal  = np.tanh( fct * alpha )

                tanValInv = 0
                if tanValInv != 0:
                    tanValInv = 1.0 / tanValInv

                tmp1    = fct * ( 1.0 - tanVal**2 ) * tanValInv
                tmp2    = tanValInv**2 * ( sol[m][i] - actSol[m][i] )**2
                tmpVal += tmp1 * ( 1.0 - tmp2 )

            return tmpVal

        for m in range( nDims ):
            tmpFunc  = lambda alpha : funcVal( alpha, m, sol, actSol, bcTime, times, nTimes )
            coefs[m] = fsolve( tmpFunc, 1.0 ) 

        return coefs
  
    def getMeanErrs( self, rType = 'trn', vType = 'vel' ):

        nDims     = self.nDims
        velNames  = self.velNames
        varNames  = self.varNames
        errVec    = np.zeros( shape = ( nDims ), dtype = 'd' )

        if rType == 'trn':
            actSol  = self.actSol
            odeObj  = self.getSol( self.params )
            trnFlag = True
        elif rType == 'oos':
            actSol  = self.actOosSol
            odeObj  = self.getOosSol()
            trnFlag = False
        else:
            assert False, 'rType is either trn or oos!'

        if odeObj is None:
            return [ np.inf ] * nDims

        sol     = odeObj.getSol()
    
        for m in range( nDims ):

            ( slope, intercept ) = self.deNormHash[ velNames[m] ]
            invFunc = lambda x : slope * x + intercept
            y       = invFunc( sol[m]    )
            yAct    = invFunc( actSol[m] )

            if vType == 'var':
                y    = self.intgVel( y,    m, trnFlag )
                yAct = self.intgVel( yAct, m, trnFlag )

            tmp1 = np.linalg.norm( y - yAct )
            tmp2 = np.linalg.norm( yAct )
                
            if tmp2 > 0:
                tmp2 = 1.0 / tmp2

            errVec[m] = tmp1 * tmp2

        return errVec

    def lighten( self ):
        
        self.actSol    = None
        self.actOosSol = None
        self.tmpVec    = None
        
    def save( self, outModFile ):

        self.logger.info( 'Saving the model to %s', 
                          outModFile  )
        
        t0 = time.time()
        
        with open( outModFile, 'wb' ) as fHd:
            dill.dump( self, fHd, pk.HIGHEST_PROTOCOL )

        self.logger.info( 'Saving the model took %0.2f seconds', 
                          time.time() - t0  )
        
    def saveGamma( self, outGammaFile ):

        with open( outGammaFile, 'wb' ) as fHd:
            pk.dump( self.Gamma, fHd, pk.HIGHEST_PROTOCOL )

    def getResiduals( self ):
        
        nDims    = self.nDims
        velNames = self.velNames
        odeObj   = self.getSol( self.params )
        sol      = odeObj.getSol()
        actSol   = self.actSol

        resVec = []
        for m in range( nDims ):
            velName              = velNames[m]
            y                    = sol[m]
            yAct                 = actSol[m]
            ( slope, intercept ) = self.deNormHash[ velName ]
            invFunc              = lambda x : slope * x + intercept
            y                    = invFunc( y )
            yAct                 = invFunc( yAct )
            res                  = y - yAct
            resVec.append( res )
        
        return resVec

    def getResCovariance( self, normFlag = False ):
        
        resVec = self.getResiduals()
        res    = np.array( resVec )
        covMat = np.cov( res )

        if not normFlag:
            return covMat

        nDims = self.nDims
        for a in range( nDims ):
            for b in range( nDims ):

                fctA = np.mean( resVec[a]**2 )
                fctB = np.mean( resVec[b]**2 )

                if fctA > 0:
                    fctA = 1.0 / np.sqrt( fctA )

                if fctB > 0:
                    fctB = 1.0 / np.sqrt( fctB )

                covMat[a][b] *= fctA * fctB

        return covMat

    def getTimeDf( self ):
        
        statHash     = self.statHash
        df           = pd.DataFrame()
        df[ 'Time' ] = [ statHash[ 'odeTime' ], 
                         statHash[ 'adjOdeTime' ] , 
                         statHash[ 'totTime' ]      ] 
        df[ 'Time' ] = df.Time.apply( lambda x : round( float(x), 2 ) )

        df[ 'Perc' ] = [ statHash[ 'odeTime' ] / statHash[ 'totTime' ],
                         statHash[ 'adjOdeTime' ] / statHash[ 'totTime' ],
                         1.0 ]
        df[ 'Perc' ] = df.Perc.apply( lambda x : str( round( 100.0 * float(x), 1 ) ) + '%' )

        df[ 'Cnt' ]  = [ statHash[ 'odeCnt' ], 
                         statHash[ 'adjOdeCnt' ], 
                         1 ]  

        df.index     = [ 'Geodesic ODE', 
                         'Adjoint ODE', 
                         'Total' ]
        return df

    def pltConv( self ):

        y     = np.log10( np.array( self.errVec ) )
        nItrs = len( y )
        x     = np.arange( 1, nItrs + 1 )

        plt.plot( x, y, 'g' )
        plt.xlabel( 'Iterations' )
        plt.ylabel( 'Log Relative Err' )
        plt.show()
        
    def setBcs( self ):
        pass

    def setActs( self ):
        pass

    def getSol( self, params ):
        pass

    def getAdjSol( self, params):
        pass

    def getGrad( self, params):
        pass
