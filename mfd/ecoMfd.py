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
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

sys.path.append( os.path.abspath( '../' ) )

from mfd.ecoMfdBase import EcoMfdCBase
from ode.odeGeo import OdeGeoConst 
from ode.odeGeo import OdeAdjConst 

# ***********************************************************************
# Some defintions
# ***********************************************************************

GEO_TOL = 1.0e-2
ADJ_TOL = 1.0e-2

# ***********************************************************************
# Class EcoMfdConst: Constant curv. manifold 
# ***********************************************************************

class EcoMfdConst( EcoMfdCBase ):

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
                    diagFlag     = True,
                    srelFlag     = False,                    
                    endBcFlag    = True,
                    varCoefs     = None,
                    srcCoefs     = None,
                    srcTerm      = None,
                    atnFct       = 1.0,
                    mode         = 'intraday',
                    logFileName  = None,                    
                    verbose      = 1     ):

        EcoMfdCBase.__init__(  self,
                               varNames     = varNames,
                               velNames     = velNames,
                               dateName     = dateName, 
                               dfFile       = dfFile,
                               minTrnDate   = minTrnDate, 
                               maxTrnDate   = maxTrnDate,
                               maxOosDate   = maxOosDate,
                               trmFuncDict  = trmFuncDict,
                               optType      = optType,
                               maxOptItrs   = maxOptItrs, 
                               optGTol      = optGTol,
                               optFTol      = optFTol,
                               stepSize     = stepSize,
                               factor       = factor,
                               regCoef      = regCoef,
                               regL1Wt      = regL1Wt,
                               nPca         = nPca,
                               endBcFlag    = endBcFlag,
                               varCoefs     = varCoefs,
                               srcCoefs     = srcCoefs,
                               srcTerm      = srcTerm,
                               atnFct       = atnFct,
                               mode         = mode,
                               logFileName  = logFileName,                               
                               verbose      = verbose     )

        self.diagFlag    = diagFlag
        self.srelFlag    = srelFlag        

        nDims            = self.nDims

        if diagFlag:
            self.nParams = nDims * ( 2 * nDims - 1 ) 
        else:
            self.nParams = int( nDims * nDims * ( nDims + 1 ) / 2 )

        if srelFlag:
            self.nParams -= nDims

        self.nParams += nDims
        
        self.params = np.zeros(shape=(self.nParams), dtype = 'd')

        self.setBcs()
        self.setActs()

        self.trnDf = None
        self.oosDf = None

    def setBcs( self ):

        nDims    = self.nDims
        nSteps   = self.nSteps
        varNames = self.varNames
        velNames = self.velNames
        trnDf    = self.trnDf

        nGammaVec = self.nParams - nDims
        
        for m in range( nDims ):
            varName            = varNames[m]
            velName            = velNames[m]

            self.endSol[m]     = list( trnDf[velName] )[nSteps]

            if self.endBcFlag:
                self.params[m + nGammaVec] = self.endSol[m]
            else:
                self.params[m + nGammaVec] = list( trnDf[velName] )[0]                

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

    def getGrad( self, params ):

        self.statHash[ 'gradCnt' ] += 1

        nDims    = self.nDims
        nTimes   = self.nTimes
        regCoef  = self.regCoef
        regL1Wt  = self.regL1Wt
        timeInc  = 1.0
        xi       = lambda a,b: 1.0 if a == b else 2.0
        grad     = np.zeros(shape=(self.nParams), dtype='d')     

        odeObj   = self.getSol(params)

        if odeObj is None:
            sys.exit()
            return None

        sol      = odeObj.getSol()

        adjOdeObj = self.getAdjSol(params, odeObj)

        if adjOdeObj is None:
            sys.exit()
            return False

        adjSol    = adjOdeObj.getSol()

        t0      = time.time()
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

                    grad[gammaId] = trapz( tmpVec, dx = timeInc ) +\
                        regCoef * ( regL1Wt * np.sign(params[gammaId] ) +\
                                    ( 1.0 - regL1Wt ) * 2.0 * params[gammaId] )    

                    gammaId += 1

        self.logger.debug( 'Setting gradient: %0.2f seconds.', 
                           time.time() - t0 )
        
        del sol
#        del adjSol
        del tmpVec
        
        gc.collect()

        nGammaVec = self.nParams - nDims
        GammaVec = params[:nGammaVec]

        tmp1 = np.linalg.norm(GammaVec)
        tmp2 = np.linalg.norm(grad)

        fct = 1.0
        if tmp2 > 0:
            fct = min(1.0, math.sqrt(abs(tmp1**2 - nGammaVec) / tmp2**2))

        for i in range(nDims):
            if self.endBcFlag:
                grad[i + nGammaVec] = adjSol[i][-1]
            else:
                grad[i + nGammaVec] = -adjSol[i][0]

        grad = fct * grad
        
        return grad

    def getSol(self, params):

        self.statHash[ 'odeCnt' ] += 1

        t0       = time.time()
        nDims    = self.nDims
        nSteps   = self.nSteps

        if self.endBcFlag:
            bcTime = nSteps   
        else:
            bcTime = 0.0

        nGammaVec = self.nParams - nDims
        GammaVec = params[:nGammaVec]
        bcVec = params[nGammaVec:]
        
        Gamma = self.getGammaArray(GammaVec)

        self.logger.debug( 'Solving geodesic...' )

        odeObj   = OdeGeoConst( Gamma    = Gamma,
                                bcVec    = bcVec,
                                bcTime   = bcTime,
                                timeInc  = 1.0,
                                nSteps   = self.nSteps,
                                intgType = 'LSODA',
                                tol      = GEO_TOL,
                                srcCoefs = self.srcCoefs,
                                srcTerm  = self.srcTerm,
                                verbose  = self.verbose       )

        sFlag = odeObj.solve()

        if not sFlag:
            self.logger.warning( 'Geodesic equation did not converge!' )
            return None
        
        self.statHash[ 'odeTime' ] += time.time() - t0

        self.logger.debug( 'Geodesic equation: %0.2f seconds.', 
                           time.time() - t0 ) 

        return odeObj

    def getAdjSol( self, params, odeObj ):

        self.statHash[ 'adjOdeCnt' ] += 1

        t0       = time.time()
        nDims    = self.nDims

        nGammaVec = self.nParams - nDims
        GammaVec = params[:nGammaVec]
        
        Gamma    = self.getGammaArray( GammaVec )
        sol      = odeObj.getSol()
        bcVec    = np.zeros( shape = ( nDims ), dtype = 'd' )
        bkFlag   = not self.endBcFlag

        self.logger.debug( 'Solving adjoint geodesic equation...' )

        adjOdeObj = OdeAdjConst( Gamma    = Gamma,
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
                                 verbose   = self.verbose       )

        sFlag  = adjOdeObj.solve()

        if not sFlag:
            self.logger.warning( 'Adjoint equation did not converge!' )
            return None

        self.statHash[ 'adjOdeTime' ] += time.time() - t0

        self.logger.debug( 'Adjoint equation: %0.2f seconds.', 
                           time.time() - t0 ) 

        return adjOdeObj

    def getGammaArray( self, GammaVec ):

        nDims   = self.nDims
        Gamma   = np.zeros( shape = ( nDims, nDims, nDims ), dtype = 'd' )
        gammaId = 0

        for m in range( nDims ):
            for a in range( nDims ):
                for b in range( a, nDims ):

                    if self.diagFlag and m != a and m != b and a != b:
                        Gamma[m][a][b] = 0.0
                        Gamma[m][b][a] = 0.0
                    elif self.srelFlag and m == a and a == b:
                        Gamma[m][a][b] = 0.0
                    else:
                        Gamma[m][a][b] = GammaVec[gammaId]
                        Gamma[m][b][a] = GammaVec[gammaId]
                        gammaId += 1

        return Gamma

    def getOosSol( self ):

        nDims     = self.nDims
        nTimes    = self.nTimes
        nOosTimes = self.nOosTimes

        nGammaVec = self.nParams - nDims
        GammaVec = self.params[:nGammaVec]        
        Gamma     = self.getGammaArray( GammaVec )
        srcCoefs  = self.srcCoefs

        self.logger.debug( 'Solving geodesic to predict...' )

        odeObj   = OdeGeoConst( Gamma    = Gamma,
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

    def pltResults( self, rType = 'all', pType = 'vel' ):

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
        odeObj    = self.getSol(self.params)
        oosOdeObj = self.getOosSol()
        sol       = odeObj.getSol()
        oosSol    = oosOdeObj.getSol()
        stdVec    = self.getConstStdVec()

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

            if pType == 'var':
                y         = self.intgVel( y,        m, True  )
                yAct      = self.intgVel( yAct,     m, True  )
                yOos      = self.intgVel( yOos,     m, False )
                yActOos   = self.intgVel( yActOos,  m, False )
                varStd    = self.intgVelStd( velStd, nTimes, True )
                varStdOos = self.intgVelStd( velStd, nOosTimes, False )
                yLow      = y    - 1.0 * varStd
                yHigh     = y    + 1.0 * varStd
                yLowOos   = yOos - 1.0 * varStdOos
                yHighOos  = yOos + 1.0 * varStdOos

            if rType == 'trn':
                plt.plot( x,    y,       'r',
                          x,    yAct,    'b'    )
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

            if pType == 'var':
                plt.ylabel( varNames[m] )
            else:
                plt.ylabel( velNames[m] )
            plt.show()

