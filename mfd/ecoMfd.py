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

from utl.utils import getLogger
from mfd.ecoMfdBase import EcoMfdCBase
from ode.odeGeo import OdeGeoConst 
from ode.odeGeo import OdeAdjConst 
from ode.odeGeo import OdeGeoConst2 
from ode.odeGeo import OdeAdjConst2 

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
            self.nGammaVec = nDims * ( 2 * nDims - 1 ) 
        else:
            self.nGammaVec = int( nDims * nDims * ( nDims + 1 ) / 2 )

        if srelFlag:
            self.nGammaVec -= nDims

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

                    if self.srelFlag and r == p and p == q:
                        continue                    

                    tmpVec  = xi(p,q) * np.multiply( sol[p][:], sol[q][:] )
                    tmpVec  = np.multiply(tmpVec, adjSol[r][:] )

                    grad[gammaId] = trapz( tmpVec, dx = timeInc ) +\
                        regCoef * ( regL1Wt * np.sign( GammaVec[gammaId] ) +\
                                    ( 1.0 - regL1Wt ) * 2.0 * GammaVec[gammaId] )    

                    gammaId += 1

        self.logger.debug( 'Setting gradient: %0.2f seconds.', 
                           time.time() - t0 )
        
        del sol
        del adjSol
        del tmpVec
        
        gc.collect()

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

        self.logger.debug( 'Solving geodesic...' )

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
            self.logger.warning( 'Geodesic equation did not converge!' )
            return None

        self.statHash[ 'odeTime' ] += time.time() - t0

        self.logger.debug( 'Geodesic equation: %0.2f seconds.', 
                           time.time() - t0 ) 

        return odeObj

    def getAdjSol( self, GammaVec, odeObj ):

        self.statHash[ 'adjOdeCnt' ] += 1

        t0       = time.time()
        nDims    = self.nDims
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

        self.logger.info( 'Solving geodesic to predict...' )

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
            self.logger.warning( 'Geodesic equation did not converge!' )
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

# ***********************************************************************
# Class EcoMfdConst2: Economic manifold ; continues adj, 2nd order
# ***********************************************************************

class EcoMfdConst2:

    def __init__( self,
                  varNames,
                  dateName, 
                  dfFile,
                  minTrnDate, 
                  maxTrnDate,
                  maxOosDate,
                  trmFuncDict  = {},
                  optType      = 'SLSQP',
                  maxOptItrs   = 100, 
                  optGTol      = 1.0e-4,
                  optFTol      = 1.0e-8,
                  stepSize     = None,
                  factor       = 4.0e-5,
                  regCoef      = 0.0,
                  regL1Wt      = 0.0,
                  diagFlag     = True,
                  srelFlag     = False,                                        
                  endBcFlag    = True,
                  nPca         = None,                    
                  varCoefs     = None,
                  srcCoefs     = None,
                  srcTerm      = None,
                  atnFct       = 1.0,
                  mode         = 'intraday',
                  avgWinSize   = 7 * 19 * 60,
                  velBcWinSize = 1 * 19 * 60,
                  logFileName  = None,
                  verbose      = 1        ):

        assert pd.to_datetime( maxOosDate ) > pd.to_datetime( maxTrnDate ),\
            'maxOosDate should be larger than maxTrnDate!'

        assert regL1Wt >= 0, 'Incorrect value; regL1Wt should be >= 0!'
        assert regL1Wt <= 1, 'Incorrect value; regL1Wt should be <= 1!'

        self.varNames    = varNames
        self.dateName    = dateName
        self.dfFile      = dfFile
        self.minTrnDate  = minTrnDate
        self.maxTrnDate  = maxTrnDate
        self.maxOosDate  = maxOosDate
        self.trmFuncDict = trmFuncDict 
        self.optType     = optType
        self.maxOptItrs  = maxOptItrs
        self.optGTol     = optGTol
        self.optFTol     = optFTol
        self.stepSize    = stepSize
        self.factor      = factor
        self.regCoef     = regCoef
        self.regL1Wt     = regL1Wt
        self.endBcFlag   = endBcFlag
        self.diagFlag    = diagFlag
        self.srelFlag    = srelFlag        
        self.mode        = mode
        self.avgWinSize  = avgWinSize
        self.velBcWinSize= velBcWinSize
        
        self.trnDf       = None
        self.oosDf       = None
        self.errVec      = []
        self.nDims       = len( varNames )
        self.statHash    = {}
        self.deNormHash  = {}
        self.verbose     = verbose
        
        self.logger = getLogger( logFileName, verbose, 'mfd' )
        
        self.statHash[ 'funCnt' ]     = 0
        self.statHash[ 'gradCnt' ]    = 0
        self.statHash[ 'odeCnt' ]     = 0
        self.statHash[ 'adjOdeCnt' ]  = 0
        self.statHash[ 'odeTime' ]    = 0.0
        self.statHash[ 'adjOdeTime' ] = 0.0
        self.statHash[ 'totTime' ]    = 0.0

        nDims = self.nDims
        
        self.pcaFlag = False

        if nPca is not None:

            self.pcaFlag = True

            if nPca == 'full':
                self.nPca = nDims
            else:
                self.nPca = nPca

            self.pca = PCA( n_components = self.nPca )

        if varCoefs is None:
            self.varCoefs = np.empty( shape = ( nDims ), dtype = 'd' )
            self.varCoefs.fill ( 1.0 )
        else:
            assert len( varCoefs ) == nDims, 'Incorrect size for varCoefs!'
            self.varCoefs = varCoefs

        self.srcCoefs = srcCoefs
        self.srcTerm  = srcTerm

        self.endVec = np.zeros( shape = ( 2 * nDims ), dtype = 'd' )
        self.bcVec  = np.zeros( shape = ( 2 * nDims ), dtype = 'd' )
        self.stdVec = np.zeros( shape = ( nDims ), dtype = 'd' )        

        self.setDf()        

        self.nSteps = self.nTimes - 1
        nTimes      = self.nTimes
        nOosTimes   = self.nOosTimes
        
        self.actSol    = np.zeros( shape = ( nDims, nTimes ),    dtype = 'd' )
        self.actOosSol = np.zeros( shape = ( nDims, nOosTimes ), dtype = 'd' )

        self.actAvgSol    = np.zeros( shape = ( nDims, nTimes ),    dtype = 'd' )
        self.actAvgOosSol = np.zeros( shape = ( nDims, nOosTimes ), dtype = 'd' )        

        self.atnCoefs = np.ones( shape = ( nTimes ) )

        self.setAtnCoefs( atnFct )

        if diagFlag:
            self.nGammaVec = nDims * ( 2 * nDims - 1 ) 
        else:
            self.nGammaVec = int( nDims * nDims * ( nDims + 1 ) / 2 )

        if srelFlag:
            self.nGammaVec -= nDims

        self.nParams = self.nGammaVec + nDims
        self.params  = np.zeros( shape = ( self.nParams ), dtype = 'd' )

        self.setBcs()
        self.setActs()

        self.trnDf = None
        self.oosDf = None
        
    def setDf( self ):

        t0         = time.time()
        dfFile     = self.dfFile
        dateName   = self.dateName
        nDims      = self.nDims
        varNames   = self.varNames
        minDt      = pd.to_datetime( self.minTrnDate )
        maxDt      = pd.to_datetime( self.maxTrnDate )
        maxOosDt   = pd.to_datetime( self.maxOosDate )
        fileExt    = dfFile.split( '.' )[-1]

        if fileExt == 'csv':
            df = pd.read_csv( dfFile ) 
        elif fileExt == 'pkl':
            df = pd.read_pickle( dfFile ) 
        else:
            assert False, 'Unknown input file extension %s' % fileExt

        if self.mode == 'day':
            tmpFunc        = lambda x : pd.to_datetime( x ).date()
            df[ 'tmp' ]    = df[ dateName ].apply( tmpFunc )
            df             = df.groupby( [ 'tmp' ] )[ varNames ].mean()
            df[ dateName ] = df.index
            df             = df.reset_index( drop = True )
        elif self.mode == 'intraday':
            pass
        else:
            assert False, 'Mode %s is not supported!' % self.mode 
        
        df             = df[ [ dateName ] + varNames ]
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
            df = self.setPcaVars( df )

        dates  = np.array( df[ dateName ] )
        nRows  = df.shape[0]
        trnCnt = 0
        for rowId in range( nRows ):
            if dates[rowId] == maxDt:
                trnCnt = rowId + 1
                break
            elif dates[rowId] > maxDt:
                trnCnt = rowId
                break

        oosCnt = nRows - trnCnt + 1

        self.trnDf = df.head( trnCnt )
        self.oosDf = df.tail( oosCnt )

        for m in range( nDims ):
            varName = varNames[m]
            avgName = varName + '_average'
            self.trnDf[ avgName ] = self.trnDf[ varName ].\
                rolling( min_periods = 1,
                         center      = False,
                         window      = self.avgWinSize ).mean()

        self.nTimes    = self.trnDf.shape[0]
        self.nOosTimes = self.oosDf.shape[0]
        self.times     = np.array( self.trnDf[ 'time' ] )
        
        self.logger.info( 'Setting data frame: %0.2f seconds', 
                          time.time() - t0  )

    def trmVars( self, df ):
 
        nDims       = self.nDims
        varNames    = self.varNames
        trmFuncDict = self.trmFuncDict

        for varId in range( nDims ):

            varName = varNames[varId]

            self.logger.debug( 'Transforming ' + varName )

            if varName in trmFuncDict.keys():
                trmFunc       = trmFuncDict[ varName ]
                df[ 'TMP' ]   = trmFunc( df[ varName ] )
                df[ 'TMP' ]   = df[ 'TMP' ].fillna( df[ varName ] )
                df[ varName ] = df[ 'TMP' ]

            fct           = self.factor
            varMax        = np.max(  df[ varName ] )
            varMin        = np.min(  df[ varName ] )
            df[ varName ] = ( df[ varName ] - varMin ) / ( varMax - varMin )
            df[ varName ] = df[ varName ] * fct

            self.deNormHash[ varName ] = ( ( varMax - varMin ) / fct, varMin )

        return df

    def setPcaVars( self, df ):

        varNames = self.varNames

        X        = np.array( df[ varNames ] )

        pcaVec   = self.pca.fit_transform( X )
        
        self.varNames = []
        for varId in range( self.nPca ):
            newVarName       = 'var_pca_' + str( varId )
            df[ newVarName ] = pcaVec[:,varId] 
            self.varNames.append( newVarName )

        self.nDims = self.nPca

        return df

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
        trnDf    = self.trnDf

        for m in range( nDims ):
            
            vec    = np.array( trnDf[varNames[m]] )
            velVec = np.gradient( vec[-self.velBcWinSize:], 2 )
            
            self.endVec[m] = vec[nSteps]
            self.endVec[m + nDims] = np.mean( velVec )

            if self.endBcFlag:
                self.bcVec[m] = self.endVec[m]
                self.bcVec[m + nDims] = self.endVec[m + nDims]                
            else:
                self.bcVec[m] = vec[0]
                self.bcVec[m + nDims] = vec[1] - vec[0]

    def setActs( self ):
        
        nDims     = self.nDims
        nOosTimes = self.nOosTimes
        varNames  = self.varNames
        trnDf     = self.trnDf
        oosDf     = self.oosDf

        for m in range( nDims ):

            varName = varNames[m]
            avgName = varName + '_average'
            
            self.actSol[m]    = np.array( trnDf[varName] )
            self.actOosSol[m] = np.array( oosDf[varName] )
            self.actAvgSol[m] = np.array( trnDf[avgName] )

            for tsId in range( nOosTimes ):
                self.actAvgOosSol[m][tsId] = self.actAvgSol[m][-1]

    def setParams( self ):

        self.logger.info( 'Running continuous adjoint optimization to set Christoffel symbols...' )

        t0 = time.time()

        if self.optType == 'GD':
            sFlag = self.setParamsGD()        
            return
        
        options  = { 'gtol'    : self.optGTol, 
                     'ftol'    : self.optFTol, 
                     'maxiter' : self.maxOptItrs, 
                     'disp'    : True              }

        bounds  = []
        for paramId in range( self.nGammaVec ):
            bounds.append( ( None, None ) )

        for paramId in range( self.nDims ):
            bounds.append( ( 0.0, 1.0e-12 ) )

        optObj = scipy.optimize.minimize( fun     = self.getObjFunc, 
                                          x0      = self.params, 
                                          method  = self.optType, 
                                          jac     = self.getGrad,
#                                          bounds  = bounds,
                                          options = options       )
        sFlag   = optObj.success
    
        self.params = optObj.x

        self.logger.info( 'Success: %s', str( sFlag ) )

        self.statHash[ 'totTime' ] = time.time() - t0

        self.logger.info( 'Setting parameters: %0.2f seconds.', 
                          time.time() - t0 )

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

            funcVal  = self.getObjFunc( self.params )

            grad     = self.getGrad( self.params )

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
        tmpVec   = np.zeros( shape = ( nTimes ), dtype = 'd' )     

        odeObj = self.getSol( params )
        
        if odeObj is None:
            return np.inf

        sol     = odeObj.getSol()

        tmpVec.fill( 0.0 )
        for varId in range( nDims ):
            tmpVec += varCoefs[varId] * atnCoefs[:] *\
                ( sol[varId] - actSol[varId] )**2 

        val = 0.5 * trapz( tmpVec, dx = 1.0 )

        tmp1  = np.linalg.norm( params, 1 )
        tmp2  = np.linalg.norm( params )
        val  += regCoef * ( regL1Wt * tmp1 + ( 1.0 - regL1Wt ) * tmp2**2 )

        del sol
        del tmpVec
        
        gc.collect()

        return val

    def getGrad( self, params ):

        self.statHash[ 'gradCnt' ] += 1

        nDims     = self.nDims
        nTimes    = self.nTimes
        regCoef   = self.regCoef
        regL1Wt   = self.regL1Wt
        actAvgSol = self.actAvgSol
        timeInc   = 1.0
        xi        = lambda a,b: 1.0 if a == b else 2.0
        grad      = np.zeros( shape = ( self.nParams ), dtype = 'd' )     

        odeObj = self.getSol( params )

        if odeObj is None:
            sys.exit()
            return None

        sol = odeObj.getSol()

        adjOdeObj = self.getAdjSol( params, odeObj )
        
        if adjOdeObj is None:
            sys.exit()
            return False

        adjSol = adjOdeObj.getSol()

        t0 = time.time()

        GammaVec = self.getGammaVec( params )
        beta     = self.getBeta( params )

        gammaId = 0
        for r in range( nDims ):
            for p in range( nDims ):
                for q in range( p, nDims ):

                    if self.diagFlag and r != p and r != q and p != q:
                        continue

                    if self.srelFlag and r == p and p == q:
                        continue                    

                    tmpVec  = xi(p,q) * np.multiply( sol[p], sol[q] )
                    tmpVec  = np.multiply(tmpVec, adjSol[r] )

                    grad[gammaId] = trapz( tmpVec, dx = timeInc ) +\
                        regCoef * ( regL1Wt * np.sign( GammaVec[gammaId] ) +\
                                    ( 1.0 - regL1Wt ) * 2.0 * GammaVec[gammaId] )    

                    gammaId += 1

        betaId = 0
        for r in range( nDims ):
            
            tmpVec = np.multiply( adjSol[r], sol[r] - actAvgSol[r] )
            
            grad[betaId + self.nGammaVec] = trapz( tmpVec, dx = timeInc ) +\
                regCoef * ( regL1Wt * np.sign( beta[betaId] ) +\
                            ( 1.0 - regL1Wt ) * 2.0 * beta[betaId] )    
            
            betaId += 1

        self.logger.debug( 'Setting gradient: %0.2f seconds.', 
                           time.time() - t0 )

        print( 'grad:', grad )
        
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

        Gamma = self.getGamma( params )
        beta  = self.getBeta( params )

        self.logger.debug( 'Solving geodesic...' )

        odeObj = OdeGeoConst2( Gamma     = Gamma,
                               beta      = beta,
                               bcVec     = self.bcVec,
                               bcTime    = bcTime,
                               timeInc   = 1.0,
                               nSteps    = self.nSteps,
                               intgType  = 'LSODA',
                               tol       = GEO_TOL,
                               actAvgSol = self.actAvgSol,                               
                               srcCoefs  = self.srcCoefs,
                               srcTerm   = self.srcTerm,
                               verbose   = self.verbose       )

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
        Gamma    = self.getGamma( params )
        beta     = self.getBeta( params )
        sol      = odeObj.getSol()
        vel      = odeObj.getVel()
        acl      = odeObj.getAcl()

        bcVec    = np.zeros( shape = ( 2 * nDims ), dtype = 'd' )

        self.logger.debug( 'Solving adjoint geodesic equation...' )

        adjOdeObj = OdeAdjConst2( Gamma     = Gamma,
                                  beta      = beta,
                                  bcVec     = bcVec,
                                  bcTime    = 0.0,
                                  timeInc   = 1.0,
                                  nSteps    = self.nSteps,
                                  intgType  = 'RK45',
                                  actSol    = self.actSol,
                                  adjSol    = sol,
                                  adjVel    = vel,
                                  adjAcl    = acl,                                  
                                  tol       = ADJ_TOL,
                                  varCoefs  = self.varCoefs,
                                  atnCoefs  = self.atnCoefs,
                                  verbose   = self.verbose       )

        sFlag = adjOdeObj.solve()

        if not sFlag:
            self.logger.warning( 'Adjoint equation did not converge!' )
            return None

        self.statHash[ 'adjOdeTime' ] += time.time() - t0

        self.logger.debug( 'Adjoint equation: %0.2f seconds.', 
                           time.time() - t0 ) 

        return adjOdeObj

    def getGamma( self, params ):

        nDims   = self.nDims
        Gamma   = np.zeros( shape = ( nDims, nDims, nDims ), dtype = 'd' )

        GammaVec = self.getGammaVec( params )
        
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

    def getGammaVec( self, params ):

        return params[:self.nGammaVec]

    def getBeta( self, params ):

        nDims   = self.nDims
        startId = self.nGammaVec
        endId   = startId + nDims
        
        return params[startId:endId]

    def getOosSol( self ):

        nDims     = self.nDims
        nTimes    = self.nTimes
        nOosTimes = self.nOosTimes
        Gamma     = self.getGamma( self.params )
        beta      = self.getBeta( self.params )
        srcCoefs  = self.srcCoefs

        self.logger.debug( 'Solving geodesic to predict...' )

        odeObj   = OdeGeoConst2( Gamma    = Gamma,
                                 beta     = beta,
                                 bcVec    = self.endVec,
                                 bcTime   = 0.0,
                                 timeInc  = 1.0,
                                 nSteps   = nOosTimes - 1,
                                 intgType = 'LSODA',
                                 tol      = GEO_TOL,
                                 actAvgSol = self.actAvgOosSol,                                                                
                                 srcCoefs = srcCoefs,
                                 srcTerm  = None,
                                 verbose  = self.verbose )

        sFlag = odeObj.solve()

        if not sFlag:
            self.logger.warning( 'Geodesic equation did not converge!' )
            return None

        return odeObj
    
    def setConstStdVec( self ): 

        nDims  = self.nDims
        actSol = self.actSol
        odeObj = self.getSol( self.params )
        sol    = odeObj.getSol()

        for varId in range( nDims ):
            tmpVec = ( sol[varId][:] - actSol[varId][:] )**2
            self.stdVec[varId] = np.sqrt( np.mean( tmpVec ) )
    
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

        odeObj = self.getSol( self.params )
        
        if odeObj is None:
            return -np.inf

        sol = odeObj.getSol()

        tmpVec.fill( 0.0 )
        for varId in range( nDims ):

            varName = self.varNames[varId]

            if varName not in varNames:
                continue

            tmpVec += varCoefs[varId] * atnCoefs[:] *\
                ( sol[varId][:] - actSol[varId][:] )**2 

        funcVal = 0.5 * trapz( tmpVec, dx = 1.0 )

        tmpVal = np.sqrt( funcVal * funcValFct )

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

        oosOdeObj = self.getOosSol()

        if oosOdeObj is None:
            return -np.inf

        oosSol = oosOdeObj.getSol()

        tmpVec.fill( 0.0 )
        for varId in range( nDims ):

            varName = self.varNames[varId]

            if varName not in varNames:
                continue

            tmpVec += varCoefs[varId] * ( oosSol[varId][:] - actOosSol[varId][:] )**2 

        funcVal = 0.5 * trapz( tmpVec, dx = 1.0 )

        tmpVal = np.sqrt( funcVal * funcValFct )

        return tmpVal

    def getOosTrendCnt( self ): 

        nDims = self.nDims
        perfs = self.getOosTrendPerfs()
        cnt   = 0            

        for varId in range( nDims ):
            if perfs[varId]:
                cnt += 1

        assert cnt <= nDims, 'Count cannot be more than nDims!' 

        ratio = float( cnt ) / nDims
               
        return ratio

    def getOosTrendPerfs( self ): 

        nDims     = self.nDims
        nOosTimes = self.nOosTimes
        actOosSol = self.actOosSol

        oosOdeObj = self.getOosSol()

        if oosOdeObj is None:
            return -np.inf

        oosSol = oosOdeObj.getSol()

        perfs = []

        for varId in range( nDims ):
            yAct     = actOosSol[varId]
            y        = oosSol[varId]
            actTrend = np.mean( yAct ) - actOosSol[varId][0]
            prdTrend = np.mean( y ) - oosSol[varId][0]
            fct      = actTrend * prdTrend
            
            if fct > 0:
                perfs.append( True )
            else:
                perfs.append( False )
            
        return perfs

    def getConstStdVec( self ): 

        return self.stdVec

    def lighten( self ):
        
        self.actSol    = None
        self.actOosSol = None
        
    def save( self, outModFile ):

        self.logger.info( 'Saving the model to %s', 
                          outModFile  )
        
        t0 = time.time()
        
        with open( outModFile, 'wb' ) as fHd:
            dill.dump( self, fHd, pk.HIGHEST_PROTOCOL )

        self.logger.info( 'Saving the model took %0.2f seconds', 
                          time.time() - t0  )
        
    def getResiduals( self ):
        
        nDims    = self.nDims
        varNames = self.varNames
        odeObj   = self.getSol( self.params )
        sol      = odeObj.getSol()
        actSol   = self.actSol

        resVec = []
        for m in range( nDims ):
            varName              = varNames[m]
            y                    = sol[m]
            yAct                 = actSol[m]
            ( slope, intercept ) = self.deNormHash[ varName ]
            invFunc              = lambda x : slope * x + intercept
            y                    = invFunc( y )
            yAct                 = invFunc( yAct )
            res                  = y - yAct
            resVec.append( res )
        
        return resVec

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

    def pltResults( self, rType = 'all' ):

        nTimes    = self.nTimes
        nOosTimes = self.nOosTimes
        nSteps    = self.nSteps
        nDims     = self.nDims
        varNames  = self.varNames
        actSol    = self.actSol
        actOosSol = self.actOosSol
        x         = np.linspace( 0, nSteps, nTimes )
        xOos      = np.linspace( nSteps, nSteps + nOosTimes-1, nOosTimes )
        odeObj    = self.getSol( self.params )
        oosOdeObj = self.getOosSol()
        sol       = odeObj.getSol()
        oosSol    = oosOdeObj.getSol()
        stdVec    = self.getConstStdVec()

        for m in range( nDims ):

            varName  = varNames[m]
            y        = sol[m]
            yAct     = actSol[m]
            yOos     = oosSol[m]
            yActOos  = actOosSol[m]

            ( slope, intercept ) = self.deNormHash[ varName ]
            invFunc  = lambda x : slope * x + intercept
            y        = invFunc( y )
            yAct     = invFunc( yAct )
            yOos     = invFunc( yOos )
            yActOos  = invFunc( yActOos )
            
            varStd   = slope * stdVec[m]
            yLow     = y    - 1.0 * varStd
            yHigh    = y    + 1.0 * varStd
            yLowOos  = yOos - 1.0 * varStd
            yHighOos = yOos + 1.0 * varStd

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

            plt.ylabel( varNames[m] )
            plt.show()
