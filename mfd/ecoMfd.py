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
from ode.odeGeo import OdeGeoExp 
from ode.odeGeo import OdeAdjExp 

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
                    stepSize     = 1.0,
                    regCoef      = 0.0,
                    regL1Wt      = 0.0,
                    nPca         = None,
                    diagFlag     = True,
                    endBcFlag    = True,
                    varCoefs     = None,
                    srcCoefs     = None,
                    srcTerm      = None,
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
                               regCoef      = regCoef,
                               regL1Wt      = regL1Wt,
                               nPca         = nPca,
                               endBcFlag    = endBcFlag,
                               varCoefs     = varCoefs,
                               srcCoefs     = srcCoefs,
                               srcTerm      = srcTerm,
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
        trnDf    = self.trnDf
        times    = np.array( trnDf.time )

        if self.endBcFlag:
            bcTime = times[nSteps]   
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
                                intgType = 'vode',
                                tol      = 1.0e-6,
                                nMaxItrs = 1000,
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

        bkFlag   = not self.endBcFlag

        if self.verbose > 1:
            print( '\nSolving adjoint geodesic equation...\n' )
    
        adjOdeObj = OdeAdjConst( Gamma    = Gamma,
                                 actSol   = self.actSol,
                                 adjSol   = sol,
                                 varCoefs = self.varCoefs,
                                 nDims    = self.nDims,
                                 nSteps   = self.nSteps,
                                 timeInc  = 1.0,
                                 bkFlag   = bkFlag,
                                 verbose  = self.verbose )

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
                                intgType = 'vode',
                                tol      = 1.0e-6,
                                nMaxItrs = 1000,
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

        nDims       = self.nDims
        nSteps      = self.nSteps
        dateName    = self.dateName
        trnDf       = self.trnDf
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
            begDate = list( trnDf[dateName] )[nSteps]

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
                                intgType = 'vode',
                                tol      = 1.0e-6,
                                nMaxItrs = 1000,
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
        trnDf     = self.trnDf
        oosDf     = self.oosDf
        varNames  = self.varNames
        velNames  = self.velNames
        actSol    = self.actSol
        actOosSol = self.actOosSol
        x         = np.array( trnDf[ 'Date' ] )
        xOos      = np.array( oosDf[ 'Date' ] )
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
            yLow     = y    - 1.0 * stdVec[m] 
            yHigh    = y    + 1.0 * stdVec[m] 
            yLowOos  = yOos - 1.0 * stdVec[m] 
            yHighOos = yOos + 1.0 * stdVec[m] 

            ( slope, intercept ) = self.deNormHash[ velName ]
            invFunc  = lambda x : slope * x + intercept
            y        = invFunc( y )
            yAct     = invFunc( yAct )
            yLow     = invFunc( yLow )
            yHigh    = invFunc( yHigh )
            yOos     = invFunc( yOos )
            yActOos  = invFunc( yActOos )
            yLowOos  = invFunc( yLowOos )
            yHighOos = invFunc( yHighOos )

            if pType == 'var':
                y        = self.intgVel( y,        m, True  )
                yAct     = self.intgVel( yAct,     m, True  )
                yLow     = self.intgVel( yLow,     m, True  )
                yHigh    = self.intgVel( yHigh,    m, True  )
                yOos     = self.intgVel( yOos,     m, False )
                yActOos  = self.intgVel( yActOos,  m, False )
                yLowOos  = self.intgVel( yLowOos,  m, False )
                yHighOos = self.intgVel( yHighOos, m, False )

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

            plt.xlabel( 'Date' )

            if pType == 'var':
                plt.ylabel( varNames[m] )
            else:
                plt.ylabel( velNames[m] )
            plt.show()

# ***********************************************************************
# Class EcoMfdExp: Manifolf with exp. metric
# ***********************************************************************

class EcoMfdExp( EcoMfdCBase ):

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
                    stepSize     = 1.0,
                    regCoef      = 0.0,
                    regL1Wt      = 0.0,
                    nPca         = None,
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
                               regCoef      = regCoef,
                               regL1Wt      = regL1Wt,
                               nPca         = nPca,
                               verbose      = verbose     )

        nDims            = self.nDims
        nTimes           = self.nTimes
        nOosTimes        = self.nOosTimes
        self.initVec     = np.zeros( shape = ( 2 * nDims ),        dtype = 'd' )    
        self.initAdjVec  = np.zeros( shape = ( 2 * nDims ),        dtype = 'd' ) 
        self.actVel      = np.zeros( shape = ( nDims, nTimes ),    dtype = 'd'  )
        self.actOosVel   = np.zeros( shape = ( nDims, nOosTimes ), dtype = 'd' )
        self.nGammaVec   = nDims * nDims
        self.GammaVec    = np.zeros( shape = ( self.nGammaVec ),   dtype = 'd' ) 

        self.setBcs()
        self.setActs()

    def trmVars( self, df ):
 
        nDims          = self.nDims
        varNames       = self.varNames
        velNames       = self.velNames
        trmFuncDict    = self.trmFuncDict

        for varId in range( nDims ):

            var    = varNames[varId]
            varVel = velNames[varId]

            if self.verbose:
                print( 'Transforming variable ' + var )

            if var in trmFuncDict.keys():
                trmFunc   = trmFuncDict[ var ]
                df[ var ] = trmFunc( df[ var ] )

            fct          = 1.0e-3
            varMax       = np.max(  df[ var ] )
            varMin       = np.min(  df[ var ] )
            fct          = fct / ( varMax - varMin )
            df[ var ]    = ( df[ var ] - varMin ) * fct
            df[ varVel ] = df[ varVel ]  * fct

        return df

    def setPcaVars( self, df ):

        assert False, 'This is not implemented yet!'

        return df

    def setBcs( self ):

        nDims    = self.nDims
        nSteps   = self.nSteps
        varNames = self.varNames
        velNames = self.velNames
        trnDf    = self.trnDf

        for m in range( nDims ):
            varName                    = varNames[m]
            velName                    = velNames[m]
            self.initVec[m]            = list( trnDf[varName] )[nSteps]
            self.initVec[m + nDims]    = list( trnDf[velName] )[nSteps]
            self.initAdjVec[m]         = 0.0
            self.initAdjVec[m + nDims] = 0.0

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

            varVec   = np.array( trnDf[varName] )
            velVec   = np.array( trnDf[velName] )
            for tsId in range( nTimes ):
                self.actSol[varId][tsId] = varVec[tsId] 
                self.actVel[varId][tsId] = velVec[tsId] 

            oosVarVec = np.array( oosDf[varName] )
            oosVelVec = np.array( oosDf[velName] )
            for tsId in range( nOosTimes ):
                self.actOosSol[varId][tsId] = oosVarVec[tsId]
                self.actOosVel[varId][tsId] = oosVelVec[tsId]

    def getGrad( self, GammaVec ):

        self.statHash[ 'gradCnt' ] += 1
 
        nDims      = self.nDims
        nTimes     = self.nTimes
        regCoef    = self.regCoef
        regL1Wt    = self.regL1Wt
        timeInc    = 1.0
        GammaCoefs = self.getGammaCoefs( GammaVec )
        grad       = np.zeros( shape = ( self.nGammaVec ), dtype = 'd' )     

        odeObj     = self.getSol( GammaVec )

        if odeObj is None:
            sys.exit()
            return None

        sol        = odeObj.getSol()
        vel        = odeObj.getVel()

        adjOdeObj  = self.getAdjSol( GammaVec, odeObj )

        if adjOdeObj is None:
            sys.exit()
            return False

        adjSol     = adjOdeObj.getSol()
        adjVel     = adjOdeObj.getVel()

        tmpSol    = np.zeros( shape = ( nDims  ), dtype = 'd' )
        tmpVec    = np.zeros( shape = ( nTimes ), dtype = 'd' )
        t0        = time.time()
        gammaId   = 0
        for p in range( nDims ):
            for q in range( nDims ):
                for tsId in range( nTimes ):
                    for a in range( nDims ):
                        tmpSol[a] = sol[a][tsId]
                    expTermpq    = np.dot( GammaCoefs[p][:] - GammaCoefs[q][:], tmpSol )
                    expTermpq    = np.exp( expTermpq )
                    tmpVec[tsId] = vel[p][tsId] * vel[q][tsId] * adjVel[p][tsId] -\
                        0.5 * expTermpq * vel[p][tsId]**2 * adjSol[q][tsId] 

                    for m in range( nDims ):
                        expTermpm    = np.dot( GammaCoefs[p][:] - GammaCoefs[m][:], tmpSol )
                        expTermpm    = np.exp( expTermpm )
                        expTermmp    = expTermpm
                        if expTermmp > 0:
                            expTermmp = 1.0 / expTermmp

                        tmpVec[tsId] = tmpVec[tsId] -\
                            0.5 * GammaCoefs[p][m] * expTermpm *\
                            sol[q][tsId] * vel[p][tsId]**2 * adjSol[m][tsId] +\
                            0.5 * GammaCoefs[m][p] * expTermmp *\
                            sol[q][tsId] * vel[m][tsId]**2 * adjSol[p][tsId]

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

        t0         = time.time()
        nDims      = self.nDims
        nSteps     = self.nSteps
        trnDf      = self.trnDf
        times      = np.array( trnDf.time )
        bcTime     = times[nSteps]   
        GammaCoefs = self.getGammaCoefs( GammaVec )

        if self.verbose > 1:
            print( '\nSolving geodesic...\n' )

        odeObj   = OdeGeoExp(   GammaCoefs = GammaCoefs,
                                bcVec      = self.initVec,
                                bcTime     = bcTime,
                                timeInc    = 1.0,
                                nSteps     = self.nSteps,
                                intgType   = 'vode',
                                tol        = 1.0e-6,
                                nMaxItrs   = 1000,
                                verbose    = self.verbose       )

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

        t0         = time.time()
        nDims      = self.nDims
        GammaCoefs = self.getGammaCoefs( GammaVec )

        if self.verbose > 1:
            print( '\nSolving adjoint geodesic equation...\n' )

        adjOdeObj = OdeAdjExp( GammaCoefs = GammaCoefs,
                               actSol     = self.actSol,
                               adjObj     = odeObj,
                               nDims      = self.nDims,
                               nSteps     = self.nSteps,
                               timeInc    = 1.0,
                               verbose    = self.verbose )

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

    def getGammaCoefs( self, GammaVec ):

        nDims      = self.nDims
        GammaCoefs = np.zeros( shape = ( nDims, nDims ), dtype = 'd' )
        gammaId    = 0

        for m in range( nDims ):
            for s in range( nDims ):
                GammaCoefs[m][s] = GammaVec[gammaId]
                gammaId += 1

        return GammaCoefs

    def getOosSol( self ):

        nOosTimes  = self.nOosTimes
        GammaCoefs = self.getGammaCoefs( self.GammaVec )
        
        if self.verbose > 0:
            print( '\nSolving geodesic to predict...\n' )

        odeObj   = OdeGeoExp(   GammaCoefs = GammaCoefs,
                                bcVec      = self.initVec,
                                bcTime     = 0.0,
                                timeInc    = 1.0,
                                nSteps     = nOosTimes - 1,
                                intgType   = 'vode',
                                tol        = 1.0e-6,
                                nMaxItrs   = 1000,
                                verbose    = self.verbose       )

        sFlag = odeObj.solve()

        if not sFlag:
            if self.verbose > 0:
                print( 'Geodesic equation did not converge!' )
            return None

        return odeObj

    def pltResults( self, velFlag = True ):

        nTimes    = self.nTimes
        nOosTimes = self.nOosTimes
        nSteps    = self.nSteps
        nDims     = self.nDims
        trnDf     = self.trnDf
        oosDf     = self.oosDf
        varNames  = self.varNames
        velNames  = self.velNames
        actSol    = self.actSol
        actOosSol = self.actOosSol
        x         = np.array( trnDf[ 'Date' ] )
        xOos      = np.array( oosDf[ 'Date' ] )
        odeObj    = self.getSol( self.GammaVec )
        oosOdeObj = self.getOosSol()
        sol       = odeObj.getSol()
        oosSol    = oosOdeObj.getSol()

        if velFlag:
            actVel    = self.actVel
            actOosVel = self.actOosVel
            vel       = odeObj.getVel()
            oosVel    = oosOdeObj.getVel()
            
        for m in range( nDims ):

            if velFlag:
                y       = vel[m]
                yAct    = actVel[m]
                yOos    = oosVel[m]
                yActOos = actOosVel[m]
            else:
                y       = sol[m]
                yAct    = actSol[m]
                yOos    = oosSol[m]
                yActOos = actOosSol[m]

            plt.plot( x,    y,       'r',
                      x,    yAct,    'b',
                      xOos, yOos,    'g', 
                      xOos, yActOos, 'b'   )
            plt.xlabel( 'Date' )
            if velFlag:
                plt.ylabel( velNames[m] )
            else:
                plt.ylabel( varNames[m] )
            plt.show()
