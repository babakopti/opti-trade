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
import tensorflow as tf
from tensorflow import keras


from scipy.integrate import trapz
from scipy.optimize import line_search
from scipy.optimize import fsolve
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LinearRegression

sys.path.append( os.path.abspath( '../' ) )

from ode.odeGeo import OdeGeoNN, OdeAdjNN

from utl.utils import getLogger

# ***********************************************************************
# Some parameters
# ***********************************************************************

SET_VAR_OFFSET = False
GEO_TOL = 1.0e-6
ADJ_TOL = 1.0e-6

# ***********************************************************************
# Class EcoMfdNN: Economic manifold ; continues adjoint; NN boosted
# ***********************************************************************

class EcoMfdNN:

    def __init__(   self,
                    varNames,
                    velNames,
                    dateName, 
                    dfFile,
                    minTrnDate, 
                    maxTrnDate,
                    maxOosDate,
                    hLayerSizes = [],                    
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
                    diagFlag     = True,
                    srelFlag     = False,                                        
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
        self.hLayerSizes = hLayerSizes        
        self.trmFuncDict = trmFuncDict 
        self.optType     = optType
        self.maxOptItrs  = maxOptItrs
        self.optGTol     = optGTol
        self.optFTol     = optFTol
        self.stepSize    = stepSize
        self.factor      = factor
        self.regCoef     = regCoef
        self.regL1Wt     = regL1Wt
        self.diagFlag    = diagFlag
        self.srelFlag    = srelFlag                
        self.endBcFlag   = endBcFlag
        self.mode        = mode
        self.trnDf       = None
        self.oosDf       = None
        self.nnModel     = None
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
        self.bcSol       = np.zeros( shape = ( nDims ), dtype = 'd' )
        self.stdVec      = np.zeros( shape = ( nDims ), dtype = 'd' )        
        self.varOffsets  = np.zeros( shape = ( nDims ), dtype = 'd' )

        self.setDf()        

        self.nSteps      = self.nTimes - 1
        nTimes           = self.nTimes
        nOosTimes        = self.nOosTimes
        self.actSol      = np.zeros( shape = ( nDims, nTimes ),    dtype = 'd' )
        self.actOosSol   = np.zeros( shape = ( nDims, nOosTimes ), dtype = 'd' )
        self.tmpVec      = np.zeros( shape = ( nTimes ), 	   dtype = 'd' )
        self.cumSol      = np.ones(shape=(nTimes, nDims))

        self.trnEndDate  = list( self.trnDf[ dateName ] )[-1]

        self.atnCoefs = np.ones( shape = ( nTimes ) )

        self.setAtnCoefs( atnFct )
        
        if diagFlag:
            self.nGammaVec = nDims * ( 2 * nDims - 1 ) 
        else:
            self.nGammaVec = int( nDims * nDims * ( nDims + 1 ) / 2 )

        if srelFlag:
            self.nGammaVec -= nDims

        self.setBcs()
        self.setActs()
        self.setNNModel()

        nWeights = 0
        for item in self.nnModel.get_weights():
            nWeights += item.size

        self.nParams = nWeights #self.nGammaVec

        self.params = np.zeros(shape=(self.nParams))

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

    def setBcs( self ):

        nDims    = self.nDims
        nSteps   = self.nSteps
        varNames = self.varNames
        velNames = self.velNames
        trnDf    = self.trnDf

        for m in range( nDims ):
            varName            = varNames[m]
            velName            = velNames[m]

            self.endSol[m]     = list( trnDf[velName] )[nSteps]

            if self.endBcFlag:
                self.bcSol[m]    = self.endSol[m]
            else:
                self.bcSol[m]    = list( trnDf[velName] )[0]

    def setActs( self ):
        
        nDims       = self.nDims
        varNames    = self.varNames
        velNames    = self.velNames
        trnDf       = self.trnDf
        oosDf       = self.oosDf
        nTimes      = self.nTimes
        nOosTimes   = self.nOosTimes

        for varId in range( nDims ):

            varName  = varNames[varId]
            velName  = velNames[varId]

            vec   = np.array( trnDf[velName] )
            for tsId in range( nTimes ):
                self.actSol[varId][tsId] = vec[tsId] 

            oosVec  = np.array( oosDf[velName] )
            for tsId in range( nOosTimes ):
                self.actOosSol[varId][tsId] = oosVec[tsId]

    def setNNModel(self):

        hLayerSizes = self.hLayerSizes
        
        layers = [
            keras.layers.InputLayer(1)#self.nDims) @hack@
        ]

        for layerSize in hLayerSizes:
            layers.append(keras.layers.Dense(layerSize, use_bias=True, activation="softsign"))

        layers.append(keras.layers.Dense(self.nGammaVec, use_bias=False, activation="linear", kernel_initializer="zeros"))

 #       custom_func = lambda x: 3.0 * keras.backend.tanh(x)
        
 #       layers.append(keras.layers.Activation(custom_func))
                    
        self.nnModel = keras.Sequential(layers)

    def setParams( self ):

        self.logger.info(
            'Running continuous adjoint optimization to set Christoffel symbols...'
        )

        t0 = time.time()
        sFlag = False
        
        if self.optType == 'GD':
            sFlag = self.setParamsGD()
        else:
            options  = { 'gtol'       : self.optGTol, 
                         'ftol'       : self.optFTol, 
                         'maxiter'    : self.maxOptItrs,
                         'eps'        : 0.001,
                         'disp'       : True              }

            optObj = scipy.optimize.minimize( fun      = self.getObjFunc, 
                                              x0       = self.params, 
                                              method   = "SLSQP",
                                              jac      = self.getGrad,
                                              options  = options          )
            sFlag   = optObj.success
    
            self.params = optObj.x

#            self.logger.info(optObj)
                
            self.logger.info( 'Success: %s', str( sFlag ) )

        self.statHash[ 'totTime' ] = time.time() - t0

        self.logger.info( 'Setting parameters: %0.2f seconds.', 
                          time.time() - t0   )

#        self.setConstStdVec()

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

        sol = self.getSol(params)

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

    def setCumSol(self, sol):

        for tsId in range(self.nTimes):
            for m in range(self.nDims):
                self.cumSol[tsId][m] = trapz(sol[m][:tsId+1], dx=1.0)

    def setConstStdVec( self ): 

        nDims  = self.nDims
        actSol = self.actSol
        sol  = self.getSol(self.params)

        for varId in range( nDims ):
            tmpVec = ( sol[varId][:] - actSol[varId][:] )**2
            self.stdVec[varId] = np.sqrt( np.mean( tmpVec ) )
            
    def save( self, outModFile ):

        self.logger.info( 'Saving the model to %s', 
                          outModFile  )
        
        t0 = time.time()
        
        with open( outModFile, 'wb' ) as fHd:
            dill.dump( self, fHd, pk.HIGHEST_PROTOCOL )

        self.logger.info( 'Saving the model took %0.2f seconds', 
                          time.time() - t0  )
        
    def pltConv( self ):

        y     = np.log10( np.array( self.errVec ) )
        nItrs = len( y )
        x     = np.arange( 1, nItrs + 1 )

        plt.plot( x, y, 'g' )
        plt.xlabel( 'Iterations' )
        plt.ylabel( 'Log Relative Err' )
        plt.show()

    def getGammaVecJacs(self):

        x = np.ones(shape=(self.nTimes, 1))
        inpVars = tf.Variable(x, dtype=tf.float64)        
#        inpVars = tf.Variable(self.cumSol, dtype=tf.float32) @hack@
        model = self.nnModel
        with tf.GradientTape() as tape:
            tape.watch(model.weights)
            outVars = model(inpVars)
            jacs = tape.jacobian(outVars, model.weights)

        assert len(jacs) == len(model.weights), 'Internal error!'
        
        jacVec = None
        for jac in jacs:
            
            shape = jac.numpy().shape

            assert shape[0] == self.nTimes, 'Internal error!'
            assert shape[1] == self.nGammaVec, 'Internal error!'
                
            size = shape[2]
            if len(shape) == 4:
                size *= shape[3]

            if jacVec is None:
                jacVec = jac.numpy().reshape((shape[0], shape[1], size)).copy()
            else:
                jacVec = np.concatenate(
                    [
                        jacVec,
                        jac.numpy().reshape((shape[0], shape[1], size)).copy()
                    ],
                    axis=2,
                )

        assert jacVec.shape[0] == self.nTimes, 'Internal error!'                    
        assert jacVec.shape[1] == self.nGammaVec, 'Internal error!'
        assert jacVec.shape[2] == self.nParams, 'Internal error!'            

        jacVec = np.swapaxes(jacVec, 0, 2)

        assert jacVec.shape[0] == self.nParams, 'Internal error!'        
        assert jacVec.shape[1] == self.nGammaVec, 'Internal error!'                    
        assert jacVec.shape[2] == self.nTimes, 'Internal error!'            
        
        return jacVec

    def getGrad( self, params ):

        t0 = time.time()
        
        self.statHash[ 'gradCnt' ] += 1

        nDims    = self.nDims
        nTimes   = self.nTimes
        regCoef  = self.regCoef
        regL1Wt  = self.regL1Wt
        timeInc  = 1.0
        xi       = lambda a,b: 1.0 if a == b else 2.0
        grad     = np.zeros(shape=(self.nParams), dtype = 'd')     

        sol = self.getSol(params)
        adjSol = self.getAdjSol(params, sol)

        weights = self.getNNWeights(params)
        self.nnModel.set_weights(weights=weights)

        gammaVecJacs = self.getGammaVecJacs()
        
        gammaId = 0
        for r in range( nDims ):
            for p in range( nDims ):
                for q in range( p, nDims ):

                    if self.diagFlag and r != p and r != q and p != q:
                        continue

                    if self.srelFlag and r == p and p == q:
                        continue                    

                    tmpVec  = xi(p,q) * np.multiply( sol[p][:], sol[q][:] )
                    tmpVec  = np.multiply(tmpVec, adjSol[r][:] )

                    for paramId in range(self.nParams):

                        grad[paramId] += trapz(
                            np.multiply(
                                tmpVec, gammaVecJacs[paramId][gammaId][:]
                            ),
                            dx = timeInc
                        ) + regCoef * (
                                regL1Wt * np.sign(
                                    params[paramId]
                                ) + (1.0 - regL1Wt) * 2.0 * params[paramId]
                            )    
                        
                    gammaId += 1

        self.logger.debug( 'Setting gradient: %0.2f seconds.', 
                           time.time() - t0 )
        
        del sol
        del adjSol
        del tmpVec
        
        gc.collect()

        return grad

    def getSol( self, params ):

        self.statHash[ 'odeCnt' ] += 1

        t0       = time.time()
        nDims    = self.nDims
        nSteps   = self.nSteps

        if self.endBcFlag:
            bcTime = nSteps   
        else:
            bcTime = 0.0

        Gamma = self.getGamma(params)

        self.logger.debug( 'Solving geodesic...' )
        
        odeObj = OdeGeoNN(
            Gamma    = Gamma,
            bcVec    = self.bcSol,
            bcTime   = bcTime,
            timeInc  = 1.0,
            nSteps   = self.nSteps,
            intgType = 'LSODA',
            tol      = GEO_TOL,
            srcCoefs = self.srcCoefs,
            srcTerm  = self.srcTerm,
            verbose  = self.verbose
        )

        if odeObj is None:
            sys.exit()
            return False
        
        sFlag = odeObj.solve()

        if not sFlag:
            self.logger.warning( 'Geodesic equation did not converge!' )
            return None

        self.statHash[ 'odeTime' ] += time.time() - t0

        self.logger.debug(
            'Geodesic equation: %0.2f seconds.', 
            (time.time() - t0),
        ) 

        sol = odeObj.getSol()

        #self.setCumSol(sol) @hack@
        
        return sol

    def getAdjSol( self, params, sol ):

        self.statHash[ 'adjOdeCnt' ] += 1

        t0       = time.time()
        nDims    = self.nDims
        bcVec    = np.zeros( shape = ( nDims ), dtype = 'd' )
        bkFlag   = not self.endBcFlag

        Gamma = self.getGamma(params)
        
        self.logger.debug( 'Solving adjoint geodesic equation...' )

        adjOdeObj = OdeAdjNN(
            Gamma    = Gamma,
            bcVec    = bcVec,
            bcTime    = 0.0,
            timeInc   = 1.0,
            nSteps    = self.nSteps,
            intgType  = 'RK45',
            actSol    = self.actSol,
            adjSol    = sol,
            tol       = ADJ_TOL,
            varCoefs  = self.varCoefs,
            atnCoefs  = self.atnCoefs,
            verbose   = self.verbose
        )

        if adjOdeObj is None:
            sys.exit()
            return False
        
        sFlag  = adjOdeObj.solve()

        if not sFlag:
            self.logger.warning( 'Adjoint equation did not converge!' )
            return None

        self.statHash[ 'adjOdeTime' ] += time.time() - t0

        self.logger.debug( 'Adjoint equation: %0.2f seconds.', 
                           time.time() - t0 ) 

        adjSol = adjOdeObj.getSol()
        
        return adjSol

    def getNNWeights(self, params):
        
        curWeights = self.nnModel.get_weights()

        weights = []
        begInd = 0
        endInd = 0
        for item in curWeights:
            endInd = begInd + item.size
            weights.append(
                np.array(params[begInd:endInd]).reshape(item.shape)
            )
            begInd = endInd

        return weights
                      
    def getGamma(self, params):

        nDims   = self.nDims
        params_before = params.copy()
        weights = self.getNNWeights(params)
        
        self.nnModel.set_weights(weights=weights)

#        inpVars = tf.Variable(self.cumSol, dtype=tf.float32) @hack@
        x = np.ones(shape=(self.nTimes, 1))
        inpVars = tf.Variable(x, dtype=tf.float64)

        GammaVec = self.nnModel(inpVars).numpy()

        # for tsId in range(self.nTimes):
        #     GammaVec[tsId] = params

        assert GammaVec.shape[0] == self.nTimes, 'Internal error!'
        assert GammaVec.shape[1] == self.nGammaVec, 'Internal error!'

        Gamma = np.zeros( shape = ( self.nTimes, nDims, nDims, nDims ), dtype = 'd' )
        
        gammaId = 0
        for m in range( nDims ):
            for a in range( nDims ):
                for b in range( a, nDims ):

                    if self.diagFlag and m != a and m != b and a != b:
                        continue
                    elif self.srelFlag and m == a and a == b:
                        continue
                    else:
                        for tsId in range(self.nTimes):
                            Gamma[tsId][m][a][b] = GammaVec[tsId][gammaId]
                            Gamma[tsId][m][b][a] = GammaVec[tsId][gammaId]
                        
                    gammaId += 1

        return Gamma

    def getOosSol( self, params ):

        nDims     = self.nDims
        nTimes    = self.nTimes
        nOosTimes = self.nOosTimes
        Gamma     = self.getGammaArray( self.GammaVec )
        srcCoefs  = self.srcCoefs

        self.logger.debug( 'Solving geodesic to predict...' )

        odeObj   = OdeGeoNN(    Gamma    = Gamma,
                                bcVec    = self.endSol,
                                bcTime   = 0.0,
                                timeInc  = 1.0,
                                nSteps   = nOosTimes - 1,
                                intgType = 'LSODA',
                                tol      = GEO_TOL,
                                srcCoefs = srcCoefs,
                                srcTerm  = None,
                                verbose  = self.verbose       )

        sFlag = odeObj.solve()

        if not sFlag:
            self.logger.warning( 'Geodesic equation did not converge!' )
            return None

        return odeObj

    def pltResults( self, rType = 'trn'):

        nTimes    = self.nTimes
        nOosTimes = self.nOosTimes
        nSteps    = self.nSteps
        nDims     = self.nDims
        varNames  = self.varNames
        velNames  = self.velNames
        actSol    = self.actSol
        actOosSol = self.actOosSol
        x         = np.linspace( 0, nSteps, nTimes )
        xOos      = np.linspace( nSteps, nSteps + nOosTimes-1, nOosTimes )
        sol       = self.getSol(self.params)
        oosSol    = actOosSol #self.getOosSol(self.params)
        stdVec    = self.stdVec

        for m in range( nDims ):

            velName  = velNames[m]
            y        = sol[m]
            yAct     = actSol[m]
            yOos     = oosSol[m]
            yActOos  = actOosSol[m]

            ( slope, intercept ) = self.deNormHash[ velName ]
            invFunc  = lambda x : slope * x + intercept
            y        = invFunc( y )
            yAct     = invFunc( yAct )
            yOos     = invFunc( yOos )
            yActOos  = invFunc( yActOos )
            
            velStd   = slope * stdVec[m]
            yLow     = y    - 1.0 * velStd
            yHigh    = y    + 1.0 * velStd
            yLowOos  = yOos - 1.0 * velStd
            yHighOos = yOos + 1.0 * velStd

            if rType == 'trn':
                plt.plot( x,    y,       'r-',
                          x,    yAct,    'bo'    )
                plt.fill_between( x, yLow, yHigh, alpha = 0.2 ) 

            elif rType == 'oos':
                plt.plot( xOos, yOos,    'g', 
                          xOos, yActOos, 'b'   )
                plt.fill_between( xOos, yLowOos, yHighOos, alpha = 0.2 ) 

            else:
                plt.plot( x,    y,       'r',
                          x,    yAct,    'b',
                          xOos, yOos,    'g', 
                          xOos, yActOos, 'b'   )

                plt.fill_between( x, yLow, yHigh, alpha = 0.2 ) 

                plt.fill_between( xOos, yLowOos, yHighOos, alpha = 0.2 ) 

            plt.xlabel( 'Time' )

            plt.ylabel( velNames[m] )
            
            plt.show()
