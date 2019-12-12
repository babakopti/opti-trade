# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os

sys.path.append( os.path.abspath( '../' ) )

import math
import time
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D
import pickle as pk
import dill

from scipy.integrate import trapz
from scipy.optimize import line_search
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

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
                    df,
                    minTrnDate, 
                    maxTrnDate,
                    maxOosDate,
                    trmFuncDict  = {},
                    optType      = 'L-BFGS-B',
                    maxOptItrs   = 100, 
                    optGTol      = 1.0e-4,
                    optFTol      = 1.0e-8,
                    stepSize     = 1.0,
                    factor       = 4.0e-5,
                    regCoef      = 0.0,
                    regL1Wt      = 0.0,
                    nPca         = None,
                    diagFlag     = True,
                    endBcFlag    = True,
                    varCoefs     = None,
                    srcCoefs     = None,
                    srcTerm      = None,
                    atnFct       = 1.0,
                    mode         = 'intraday',
                    verbose      = 1     ):

        EcoMfdCBase.__init__(  self,
                               varNames     = varNames,
                               velNames     = velNames,
                               dateName     = dateName, 
                               df           = df,
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
                               verbose      = verbose     )

        self.diagFlag    = diagFlag

        nDims            = self.nDims

        if diagFlag:
            self.nGammaVec = nDims * ( 2 * nDims - 1 ) 
        else:
            self.nGammaVec = int( nDims * nDims * ( nDims + 1 ) / 2 )

        self.GammaVec    = np.zeros( shape = ( self.nGammaVec ),   dtype = 'd' ) 

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

    def getGrad( self, GammaVec ):

        self.statHash[ 'gradCnt' ] += 1

        nDims    = self.nDims
        nTimes   = self.nTimes
        regCoef  = self.regCoef
        regL1Wt  = self.regL1Wt
        timeInc  = 1.0
        xi       = lambda a,b: 1.0 if a == b else 2.0
        grad     = np.zeros( shape = ( self.nGammaVec ), dtype = 'd' )     

        odeObj   = self.getSol( GammaVec )

        if odeObj is None:
            sys.exit()
            return None

        sol      = odeObj.getSol()

        adjOdeObj = self.getAdjSol( GammaVec, odeObj )

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

                    tmpVec  = xi(p,q) * np.multiply( sol[p][:], sol[q][:] )
                    tmpVec  = np.multiply(tmpVec, adjSol[r][:] )

                    grad[gammaId] = trapz( tmpVec, dx = timeInc ) +\
                        regCoef * ( regL1Wt * np.sign( GammaVec[gammaId] ) +\
                                    ( 1.0 - regL1Wt ) * 2.0 * GammaVec[gammaId] )    

                    gammaId += 1

        if self.verbose > 1:
            print( '\nSetting gradient:', 
                   round( time.time() - t0, 2 ), 
                   'seconds.\n'         )

        return grad

    def getSol( self, GammaVec ):

        self.statHash[ 'odeCnt' ] += 1

        t0       = time.time()
        nDims    = self.nDims
        nSteps   = self.nSteps

        if self.endBcFlag:
            bcTime = nSteps   
        else:
            bcTime = 0.0

        Gamma    = self.getGammaArray( GammaVec )

        if self.verbose > 1:
            print( '\nSolving geodesic...\n' )

        odeObj   = OdeGeoConst( Gamma    = Gamma,
                                bcVec    = self.bcSol,
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
            if self.verbose > 0:
                print( 'Geodesic equation did not converge!' )
            return None

        self.statHash[ 'odeTime' ] += time.time() - t0

        if self.verbose > 1:
            print( '\nGeodesic equation:', 
                   round( time.time() - t0, 2 ), 
                   'seconds.\n'         )

        return odeObj

    def getAdjSol( self, GammaVec, odeObj ):

        self.statHash[ 'adjOdeCnt' ] += 1

        t0       = time.time()
        nDims    = self.nDims
        Gamma    = self.getGammaArray( GammaVec )
        sol      = odeObj.getSol()
        bcVec    = np.zeros( shape = ( nDims ), dtype = 'd' )
        bkFlag   = not self.endBcFlag

        if self.verbose > 1:
            print( '\nSolving adjoint geodesic equation...\n' )

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
            if self.verbose > 0:
                print( 'Adjoint equation did not converge!' )
            return None

        self.statHash[ 'adjOdeTime' ] += time.time() - t0

        if self.verbose > 1:
            print( '\nAdjoint equation:', 
                   round( time.time() - t0, 2 ), 
                   'seconds.\n'         )

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
                    else:
                        Gamma[m][a][b] = GammaVec[gammaId]
                        Gamma[m][b][a] = GammaVec[gammaId]
                        gammaId += 1

        return Gamma

    def saveGamma( self, outGammaFile ):

        Gamma = self.getGammaArray( self.GammaVec )

        with open( outGammaFile, 'wb' ) as fHd:
            pk.dump( Gamma, fHd, pk.HIGHEST_PROTOCOL )

    def getOosSol( self ):

        nDims     = self.nDims
        nTimes    = self.nTimes
        nOosTimes = self.nOosTimes
        Gamma     = self.getGammaArray( self.GammaVec )
        srcCoefs  = self.srcCoefs

        if self.verbose > 0:
            print( '\nSolving geodesic to predict...\n' )

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
            if self.verbose > 0:
                print( 'Geodesic equation did not converge!' )
            return None

        return odeObj

    def predict( self, 
                 endDate,
                 begDate  = None, 
                 initVels = None    ):

        assert False, 'This routine is broken because of stdVec stuff!'

        nDims       = self.nDims
        nSteps      = self.nSteps
        dateName    = self.dateName
        velNames    = self.velNames
        varNames    = self.varNames
        Gamma       = self.getGammaArray( self.GammaVec )
        stdVec      = self.getConstStdVec()

        if initVels is not None:

            assert len( initVels ) == self.nDims, 'Incorrect size for initVels!'

            bcVec = initVels.copy()

            for m in range( nDims ):

                ( slope, intercept ) = self.deNormHash[ velNames[m] ]

                slopeInv     = slope
                if slopeInv != 0:
                    slopeInve = 1.0 / slopeInv

                bcVec[m] = slopeInv * ( bcVec[m] - intercept )
        
        else:
            bcVec = self.endSol

        if begDate is None:
            begDate = self.trnEndDate

        begDt    = pd.to_datetime( begDate )
        endDt    = pd.to_datetime( endDate )
        tmpDt    = begDt
        dates    = []
        while tmpDt <= endDt:
            if tmpDt.isoweekday() not in [ 6, 7 ]:
                dates.append( tmpDt.strftime( '%Y-%m-%d' ) )
            tmpDt = tmpDt + pd.DateOffset( days = 1 )

        nTimes = len( dates )

        if self.verbose > 0:
            print( '\nSolving geodesic to predict...\n' )

        odeObj   = OdeGeoConst( Gamma    = Gamma,
                                bcVec    = bcVec,
                                bcTime   = 0.0,
                                timeInc  = 1.0,
                                nSteps   = nTimes - 1,
                                intgType = 'LSODA',
                                tol      = GEO_TOL,
                                verbose  = self.verbose       )

        sFlag  = odeObj.solve()

        if not sFlag:
            if self.verbose > 0:
                print( 'Geodesic equation did not converge!' )
            return None

        prdSol  = odeObj.getSol()

        outHash = {}

        outHash[ dateName ] = dates

        for m in range( nDims ):
            velName    = velNames[m]
            varName    = varNames[m]
            velStdName = 'std_' + velName
            varStdName = 'std_' + varName
            ( slope, intercept )  = self.deNormHash[ velName ]
            velVec                = slope * prdSol[m] + intercept
            velStd                = slope * stdVec[m]
            varVec                = self.intgVel( velVec, m, False )
            tmpVec                = velVec + velStd
            varStdVec             = self.intgVel( tmpVec, m, False )
            varStdVec             = varStdVec - varVec
            outHash[ velName    ] = velVec
            outHash[ velStdName ] = velStd
            outHash[ varName    ] = varVec
            outHash[ varStdName ] = varStdVec

        outDf  = pd.DataFrame( outHash )

        return outDf

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
        odeObj    = self.getSol( self.GammaVec )
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

