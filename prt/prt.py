# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import dill
import time
import datetime
import talib
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

from collections import Counter
from scipy.special import erf
from scipy.optimize import minimize

sys.path.append( os.path.abspath( '../' ) )

from mod.mfdMod import MfdMod
from ode.odeGeo import OdeGeoConst 
from utl.utils import getLogger

# ***********************************************************************
# Some parameters
# ***********************************************************************

ODE_TOL   = 1.0e-2
OPT_TOL   = 1.0e-6
MAX_ITERS = 10000

# ***********************************************************************
# Class MfdPrt: Portfolio builder class
# ***********************************************************************

class MfdPrt:

    def __init__(   self,
                    modFile,
                    quoteHash,
                    nRetTimes,
                    nPrdTimes,
                    strategy     = 'mad',
                    minProbLong  = 0.5,
                    minProbShort = 0.5,
                    vType        = 'vel',
                    fallBack     = 'macd',
                    optTol       = OPT_TOL,
                    minAbsWt     = 1.0e-4,
                    logFileName  = None,                    
                    verbose      = 1          ):

        self.mfdMod       = dill.load( open( modFile, 'rb' ) ) 
        self.ecoMfd       = self.mfdMod.ecoMfd
        self.minProbLong  = minProbLong
        self.minProbShort = minProbShort
        self.fallBack     = fallBack
        self.optTol       = optTol
        self.minAbsWt     = minAbsWt
        self.logFileName  = logFileName
        self.verbose      = verbose
        self.logger       = getLogger( logFileName, verbose, 'prt' )        
        
        if vType == 'vel':
            self.vList = self.ecoMfd.velNames
        elif vType == 'var':
            self.vList = self.ecoMfd.varNames
        else:
            self.logger.error( 'Unknown vType %s', vType )
            assert False, 'Unknown vType %s' % vType

        self.vType = vType

        self.quoteHash = quoteHash
        self.assets    = []
        for asset in quoteHash:
            if asset not in self.vList:
                self.logger.warning( 'Dropping %s ; not found in the model %s',
                                     asset,
                                     modFile )
                continue

            self.assets.append( asset )

        self.nRetTimes   = nRetTimes
        self.nPrdTimes   = nPrdTimes

        if strategy not in [ 'mad', 'mean_mad', 'prd_gain_std', 'equal' ]:
            self.logger.error( 'Strategy %s is not known!', strategy )
            assert False, 'Strategy %s is not known!' % strategy

        self.strategy  = strategy

        if minProbLong <= 0:
            self.logger.error( 'minProbLong should be > 0!' )
            assert False, 'minProbLong should be > 0!'

        if minProbLong >= 1:
            self.logger.error( 'minProbLong should be < 1!' )
            assert False,  'minProbLong should be < 1!'

        if minProbShort <= 0:
            self.logger.error( 'minProbShort should be > 0!' )
            assert False, 'minProbShort should be > 0!'

        if minProbShort >= 1:
            self.logger.error( 'minProbShort should be < 1!' )
            assert False, 'minProbShort should be < 1!'

        self.retDf        = None
        self.prdSol       = None
        self.stdVec       = None
        self.optFuncVals  = None
        self.trendHash    = None

        self.setRetDf()
        self.setPrdSol()
        self.setPrdStd()
        self.setPrdTrends()

    def setRetDf( self ):

        self.retDf = pd.DataFrame()
        nRetTimes  = self.nRetTimes
        ecoMfd     = self.ecoMfd
        actSol     = ecoMfd.actSol
        nOosTimes  = ecoMfd.nOosTimes

        for m in range( ecoMfd.nDims ):
            asset     = self.vList[m]
            
            if asset not in self.assets:
                continue

            tmp       = ecoMfd.deNormHash[ asset ]
            slope     = tmp[0]
            intercept = tmp[1]
            df        = pd.DataFrame( { asset : slope * actSol[m][-nRetTimes:] +\
                                            intercept } )

            self.retDf[ asset ] = np.log( df[ asset ] ).pct_change().dropna()

    def setPrdSol( self ):

        nPrdTimes = self.nPrdTimes        
        ecoMfd    = self.ecoMfd
        nDims     = ecoMfd.nDims
        actOosSol = ecoMfd.actOosSol
        Gamma     = ecoMfd.getGammaArray( ecoMfd.GammaVec )
        bcVec     = np.zeros( shape = ( nDims ), dtype = 'd' )

        for m in range( ecoMfd.nDims ):
            
            item = self.vList[m]

            if item in self.quoteHash:
                tmp       = ecoMfd.deNormHash[ item ]
                slope     = tmp[0]
                intercept = tmp[1]
                
                slopeInv  = slope
                if slopeInv != 0:
                    slopeInv = 1.0 / slopeInv
                    
                bcVec[m] = slopeInv * ( self.quoteHash[ item ] - intercept )
            else:
                bcVec[m] = actOosSol[m][-1] 

        odeObj = OdeGeoConst( Gamma    = Gamma,
                              bcVec    = bcVec,
                              bcTime   = 0.0,
                              timeInc  = 1.0,
                              nSteps   = nPrdTimes - 1,
                              intgType = 'LSODA',
                              tol      = ODE_TOL,
                              verbose  = self.verbose       )

        sFlag    = odeObj.solve()

        if not sFlag:
            self.logger.error( 'Geodesic equation did not converge!' )
            return None

        prdSol   = odeObj.getSol()

        for m in range( ecoMfd.nDims ):
            asset     = self.vList[m]
            tmp       = ecoMfd.deNormHash[ asset ]
            slope     = tmp[0]
            intercept = tmp[1]
            
            for i in range( nPrdTimes ):
                prdSol[m][i] = slope * prdSol[m][i] + intercept

        if prdSol.shape[0] != ecoMfd.nDims:
            self.logger.error( 'Inconsistent prdSol size!' )
            assert False, 'Inconsistent prdSol size!'

        if prdSol.shape[1] <= 0:
            self.logger.error( 'Number of minutes should be positive!' )
            assert False, 'Number of minutes should be positive!'
        
        self.prdSol = prdSol

        return prdSol

    def setPrdStd( self ):

        ecoMfd = self.ecoMfd
        stdVec = ecoMfd.getConstStdVec()

        for m in range( ecoMfd.nDims ):
            asset     = self.vList[m]
            tmp       = ecoMfd.deNormHash[ asset ]
            slope     = tmp[0]
            stdVec[m] = slope * stdVec[m]

        self.stdVec = stdVec

        return stdVec

    def setPrdTrends( self ):

        mfdMod    = self.mfdMod
        ecoMfd    = self.ecoMfd
        nDims     = ecoMfd.nDims
        prdSol    = self.prdSol
        stdVec    = self.stdVec
        nPrdTimes = prdSol.shape[1]
        perfs     = ecoMfd.getOosTrendPerfs( self.vType )

        if nPrdTimes <= 0:
            self.logger.error( 'nPrdTimes should be positive!' )
            assert False, 'nPrdTimes should be positive!'

        self.trendHash = {}

        for m in range( nDims ):
            asset    = self.vList[m]

            if asset not in self.assets:
                continue

            curPrice = prdSol[m][0]
            trend    = 0.0
            prob     = 0.0

            for i in range( nPrdTimes ):

                prdPrice = prdSol[m][i]
                priceStd = stdVec[m]

                if prdPrice > curPrice:
                    fct = 1.0
                else:
                    fct = -1.0

                trend += fct

                tmp1   = curPrice - prdPrice
                tmp2   = np.sqrt( 2.0 ) * priceStd 

                if tmp2 > 0:
                    tmp2   = 1.0 / tmp2

                prob  += 0.5 * ( 1.0  - fct * erf( tmp1 * tmp2 ) )
            
            trend /= nPrdTimes
            prob  /= nPrdTimes
        
            if self.fallBack is None:
                self.trendHash[ asset ] = ( trend, prob )
            else:
                if perfs[m]: 
                    self.trendHash[ asset ] = ( trend, prob )
                else:
                    self.logger.warning( 'Falling back on %s for asset %s!',
                                         self.fallBack,
                                         asset )
                    if self.fallBack == 'sign_trick':
                        self.trendHash[ asset ] = ( -trend, 0.5 )
                    elif self.fallBack == 'macd':
                        macdTrend = self.getMacdTrend( m )
                        self.trendHash[ asset ] = ( macdTrend, 0.5 )
                    elif self.fallBack == 'msd':
                        msdTrend = self.getMsdTrend( m )
                        self.trendHash[ asset ] = ( msdTrend, 0.5 )
                    elif self.fallBack == 'zero':
                        self.trendHash[ asset ] = ( 0.0, 0.5 )

        return self.trendHash

    def getPortfolio( self ):

        if self.strategy == 'equal':
            return self.getPortfolioEq()
        else:
            return self.getPortfolioOpt()

    def getPortfolioEq( self ):

        t0           = time.time()
        strategy     = self.strategy
        assets       = self.assets
        minProbLong  = self.minProbLong 
        minProbShort = self.minProbShort
        nAssets      = len( assets )
        trendHash    = self.trendHash

        assert strategy == 'equal', 'Internal error!'
        
        assert nAssets > 0,\
            'Number of assets should be positive!'
        
        weights = np.empty( shape = ( nAssets ), dtype = 'd' )

        for i in range( nAssets ):
            asset = assets[i]
            trend = trendHash[ asset ][0]
            prob  = trendHash[ asset ][1]

            if trend > 0 and prob >= minProbLong:
                weights[i] = 1.0
            elif trend < 0 and prob >= minProbShort:
                weights[i] = -1.0
            else:
                weights[i] = 0.0
            
        sumWt = np.sum( abs( weights ) )

        sumInv = sumWt

        if sumInv > 0:
            sumInv = 1.0 / sumInv

        weights = sumInv * weights 
                
        prtHash = {}
        fct     = 0.0
        for i in range( nAssets ):
            asset = assets[i]
            wt    = weights[i]

            if abs( wt ) < self.minAbsWt:
                continue

            fct += abs( wt )
            
            prtHash[ asset ] = wt

        if fct > 0:
            fct = 1.0 / fct
        elif fct < 0:
            self.logger.error( 'Internal error: fct should be non-negative!' )
            sys.exit()
            
        for asset in prtHash:
            prtHash[ asset ] = fct * prtHash[ asset ]

        self.logger.info( 'Building portfolio took %0.2f seconds!', 
                          round( time.time() - t0, 2 ) )

        self.logger.info( 'Sum of wts: %0.4f', sum( abs( weights ) ) )

        return prtHash
    
    def getPortfolioOpt( self ):

        t0           = time.time()
        strategy     = self.strategy
        assets       = self.assets
        minProbLong  = self.minProbLong 
        minProbShort = self.minProbShort
        nAssets      = len( assets )
        trendHash    = self.trendHash

        self.optFuncVals = []
        
        optFunc   = self.getOptFunc()
        optCons   = self.getOptCons()
        guess     = self.getInitGuess()

        results   = minimize( fun         = optFunc, 
                              x0          = guess, 
                              method      = 'SLSQP',
                              tol         = self.optTol,
                              constraints = optCons,
                              options     = { 'maxiter' : MAX_ITERS } )

        self.logger.info( results[ 'message' ] )
        self.logger.info( 'Optimization success: %s',
                          str( results[ 'success' ] ) )
        self.logger.info( 'Number of function evals: %d', results[ 'nfev' ] )

        weights   = results.x

        if len( weights ) != nAssets:
            self.logger.error( 'Inconsistent size of weights!' )
            assert False, 'Inconsistent size of weights!'

        self.checkCons( optCons, weights )                   
            
        prtHash = {}
        fct     = 0.0
        for i in range( nAssets ):
            asset = assets[i]
            wt    = weights[i]

            if abs( wt ) < self.minAbsWt:
                continue

            fct += abs( wt )
            
            prtHash[ asset ] = wt

        if fct > 0:
            fct = 1.0 / fct
        elif fct < 0:
            self.logger.error( 'Internal error: fct should be non-negative!' )
            sys.exit()
            
        for asset in prtHash:
            prtHash[ asset ] = fct * prtHash[ asset ]

        self.logger.info( 'Building portfolio took %0.2f seconds!', 
                          round( time.time() - t0, 2 ) )

        self.logger.info( 'Sum of wts: %0.4f', sum( abs( weights ) ) )

        return prtHash    

    def getOptFunc( self ):

        strategy = self.strategy

        if strategy == 'mad':
            optFunc = self.getMadFunc
        elif strategy == 'mean_mad':
            optFunc = self.getMeanMadFunc
        elif strategy == 'prd_gain_std':
            optFunc = self.getPrdGainStdFunc
        else:
            self.logger.error( 'Strategy %s is not known!', strategy )
            assert False, 'Strategy %s is not known!' % strategy

        return optFunc

    def getOptCons( self ):

        strategy     = self.strategy
        minProbLong  = self.minProbLong 
        minProbShort = self.minProbShort
        assets       = self.assets
        nAssets      = len( assets )
        trendHash    = self.trendHash
        cons         = []
        sumFunc      = lambda wts : ( sum( abs( wts ) ) - 1.0 )

        cons.append( { 'type' : 'eq', 'fun' : sumFunc } )

        for i in range( nAssets ):
            asset = assets[i]
            trend = trendHash[ asset ][0]
            prob  = trendHash[ asset ][1]

            if trend > 0 and prob >= minProbLong:
                trendFunc = lambda wts : wts[i]
            elif trend < 0 and prob >= minProbShort:
                trendFunc = lambda wts : -wts[i]
            else:
                continue

            cons.append( { 'type' : 'ineq', 'fun' : trendFunc } )

        return cons

    def getInitGuess( self ):

        minProbLong  = self.minProbLong 
        minProbShort = self.minProbShort
        assets       = self.assets
        nAssets      = len( assets )
        trendHash    = self.trendHash
        guess        = np.ones( nAssets )

        for i in range( nAssets ):
            asset = assets[i]
            trend = trendHash[ asset ][0]
            prob  = trendHash[ asset ][1]

            if trend > 0 and prob >= minProbLong:
                guess[i]  = 1.0
            elif trend < 0 and prob >= minProbShort:
                guess[i]  = -1.0
            else:
                guess[i] = 0.0
 
        return guess
        
    def getMadFunc( self, wts ):

        mad = ( self.retDf - self.retDf.mean() ).dot( wts ).abs().mean()

        sumFunc = lambda wts : ( sum( abs( wts ) ) - 1.0 )

        if abs( sumFunc( wts ) ) < self.optTol:
            self.optFuncVals.append( mad )

        return mad 

    def getMeanMadFunc( self, wts ):

        mad  = self.getMadFunc( wts )
        mean = self.retDf.mean().dot( wts )
        
        madInv = mad
        if madInv > 0:
            madInv = 1.0 / madInv
            
        val = -abs( mean ) * madInv
        
        sumFunc = lambda wts : ( sum( abs( wts ) ) - 1.0 )

        if abs( sumFunc( wts ) ) < self.optTol:
            self.optFuncVals.append( val )

        return val

    def getPrdGainStdFunc( self, wts ):

        ecoMfd      = self.ecoMfd        
        quoteHash   = self.quoteHash
        assets      = self.assets
        prdSol      = self.prdSol
        stdVec      = self.stdVec
        nAssets     = len( assets )
        gain        = 0.0
        variance    = 0.0
        std         = 0.0
        
        for assetId in range( nAssets ):

            asset    = assets[assetId]
            curPrice = quoteHash[ asset ]

            if curPrice <= 0:
                self.logger.error( 'Price should be positive!' )
                assert False, 'Price should be positive!'
            
            for m in range( ecoMfd.nDims ):
                if self.vList[m] == asset:
                    break

            if m >= ecoMfd.nDims:
                self.logger.error( 'Asset %s not found in the model!', asset )
                assert False, 'Asset %s not found in the model!' % asset

            prdPrice  = prdSol[m][-1]

            curPrice  = np.log( curPrice )
            prdPrice  = np.log( prdPrice )
            gain     += wts[assetId] * ( prdPrice - curPrice ) /  curPrice
            variance += ( wts[assetId] * stdVec[m] / curPrice )**2

        std = np.sqrt( variance )
        
        stdInv = std
        if stdInv > 0:
            stdInc = 1.0 / stdInv
            
        val = -gain * stdInv 

        sumFunc = lambda wts : ( sum( abs( wts ) ) - 1.0 )

        if abs( sumFunc( wts ) ) < self.optTol:
            self.optFuncVals.append( val )

        return val

    def getActSolVec( self, m ):

        ecoMfd    = self.ecoMfd
        nDims     = ecoMfd.nDims
        nTimes    = ecoMfd.nTimes
        nOosTimes = ecoMfd.nOosTimes
        actSol    = ecoMfd.actSol
        actOosSol = ecoMfd.actOosSol
        nTmp      = nTimes + nOosTimes - 1
        tmpVec    = np.empty( shape = ( nTmp ), dtype = 'd' )

        if m >= nDims:
            self.logger.error( 'm should be smaller than nDims!' )
            assert False, 'm should be smaller than nDims!'

        asset     = self.vList[m]
        tmp       = ecoMfd.deNormHash[ asset ]
        slope     = tmp[0]
        intercept = tmp[1]

        for i in range( nTimes ):
            tmpVec[i] = slope * actSol[m][i] + intercept

        for i in range( 1, nOosTimes ):
            tmpVec[i + nTimes - 1] = slope * actOosSol[m][i] + intercept

        return tmpVec
    
    def getMacdTrend( self, m ):

        tmpVec = self.getActSolVec( m )

        macd, signal, hist = talib.MACD( tmpVec, 
                                         fastperiod   = 12, 
                                         slowperiod   = 26,
                                         signalperiod = 9  ) 

        return macd[-1] - signal[-1]

    def getMsdTrend( self, m ):

        tmpVec = self.getActSolVec( m )
        tmpVec = tmpVec[-nRetTimes:]

        mean   = tmpVec.mean()
        stddev = tmpVec.std()
 
        if tmpVec[-1] < mean - 1.75 * stddev:
            trend = 1.0
        elif tmpVec[-1] > mean + 1.75 * stddev:
            trend = -1.0
        else:
            trend = 0.0
        
        return trend

    def checkCons( self, cons, wts ):

        assets     = self.assets
        nAssets    = len( assets )
        trendHash  = self.trendHash

        for con in cons:
            conFunc = con[ 'fun' ]
            
            if con[ 'type' ] == 'eq':
                if abs( conFunc( wts ) ) > self.optTol:
                    self.logger.error( 'Equality constraint not satisfied!' )
                    assert False, 'Equality constraint not satisfied!'
            elif con[ 'type' ] == 'ineq':
                if conFunc( wts ) < -self.optTol:
                    self.logger.error( 'Inequality constraint not satisfied!' )
                    assert False, 'Inequality constraint not satisfied!'
            else:
                self.logger.error( 'Unknown constraint type!' )
                assert False, 'Unknown constraint type!'

        val = np.abs( np.sum( np.abs( wts ) ) - 1.0 )

        if val > self.optTol:
            self.logger.error( 'The weights dp not sum up to 1.0!' )
            assert False, 'The weights dp not sum up to 1.0!' 

        for i in range( nAssets ):
            asset = assets[i]
            wt    = wts[i]
            trend = trendHash[ asset ][0]
            val   = trend * wt

            if val < -self.optTol:
                self.logger.error( 'The weight %0.4f for asset %s does not match predicted trend!',
                                   wt,
                                   asset )
                assert False, 'The weight %0.4f for asset %s does not match predicted trend!' \
                    % ( wt, asset )

        self.logger.info( 'All constraints are satisfied!' )

    def pltIters( self ):
        plt.plot( self.optFuncVals, '-o' )
        plt.show()

# ***********************************************************************
# Class MfdOptionsPrt: Portfolio builder class for options
# ***********************************************************************

class MfdOptionsPrt:

    def __init__(   self,
                    modFile,
                    assetHash,
                    curDate,
                    minDate,
                    maxDate,
                    minProb,
                    rfiDaily     = 0.0,
                    tradeFee     = 0.0,
                    nDayTimes    = 1140,
                    logFileName  = None,                    
                    verbose      = 1          ):

        self.mfdMod       = dill.load( open( modFile, 'rb' ) )
        self.ecoMfd       = self.mfdMod.ecoMfd
        self.logFileName  = logFileName
        self.verbose      = verbose
        self.logger       = getLogger( logFileName, verbose, 'prt' )
        self.assetHash    = assetHash        
        self.curDate      = pd.to_datetime( curDate )
        self.minDate      = pd.to_datetime( minDate )
        self.maxDate      = pd.to_datetime( maxDate )
        self.minProb      = minProb        
        self.nDayTimes    = nDayTimes
        self.rfiDaily     = rfiDaily
        self.tradeFee     = tradeFee
        self.prdDf        = None

        setA = set( self.assetHash.keys() )
        setB = set( self.ecoMfd.velNames )
        
        assert setA == setB, 'modFile and assetHash are not consistent!'

        self.curDate = pd.to_datetime( self.curDate.strftime( '%Y-%m-%d' ) )
        self.minDate = self.minDate.replace( hour = 0,  minute = 0  )        
        self.maxDate = self.maxDate.replace( hour = 23, minute = 59 )

        if minProb <= 0 or minProb >= 1:
            self.logger.error( 'minProb should be in (0,1)!' )
            assert False, 'minProb should be in (0,1)!'

        if rfiDaily < 0 or rfiDaily > 1:
            self.logger.error( 'rfiDaily should be in [0,1]!' )
            assert False, 'rfiDaily should be in [0,1]!'
            
        self.setPrdDf()
        
    def setPrdDf( self ):

        t0        = time.time()
        ecoMfd    = self.ecoMfd
        nDims     = ecoMfd.nDims
        Gamma     = ecoMfd.getGammaArray( ecoMfd.GammaVec )
        bcVec     = np.zeros( shape = ( nDims ), dtype = 'd' )
        dateList  = []
        date      = self.curDate

        while date <= self.maxDate:
                
            while True:
                if date.isoweekday() not in [6, 7]:
                    break
                else:
                    date += datetime.timedelta( days = 1 )

            if date > self.maxDate:
                break
            
            dateStr = date.strftime( '%Y-%m-%d' )
            
            for k in range( self.nDayTimes ):
                dateList.append( dateStr )
                    
            date += datetime.timedelta( days = 1 )

        nPrdTimes = len( dateList )

        for m in range( nDims ):
            
            asset = ecoMfd.velNames[m]

            tmp       = ecoMfd.deNormHash[ asset ]
            slope     = tmp[0]
            intercept = tmp[1]
                
            slopeInv  = slope
            if slopeInv != 0:
                slopeInv = 1.0 / slopeInv

            bcVec[m] = slopeInv * ( self.assetHash[ asset ] - intercept )

        odeObj = OdeGeoConst( Gamma    = Gamma,
                              bcVec    = bcVec,
                              bcTime   = 0.0,
                              timeInc  = 1.0,
                              nSteps   = nPrdTimes - 1,
                              intgType = 'LSODA',
                              tol      = ODE_TOL,
                              verbose  = self.verbose       )

        sFlag = odeObj.solve()

        if not sFlag:
            self.logger.error( 'Geodesic equation did not converge!' )
            return None

        prdSol = odeObj.getSol()

        if prdSol.shape[0] != nDims:
            self.logger.error( 'Inconsistent prdSol size!' )
            assert False, 'Inconsistent prdSol size!'

        if prdSol.shape[1] <= 0:
            self.logger.error( 'Number of minutes should be positive!' )
            assert False, 'Number of minutes should be positive!'

        stdVec = ecoMfd.getConstStdVec()

        if stdVec.shape[0] != nDims:
            self.logger.error( 'Inconsistent stdVec size!' )
            assert False, 'Inconsistent stdVec size!'
        
        for m in range( nDims ):
            asset     = ecoMfd.velNames[m]
            tmp       = ecoMfd.deNormHash[ asset ]
            slope     = tmp[0]
            intercept = tmp[1]
            
            for i in range( nPrdTimes ):
                prdSol[m][i] = slope * prdSol[m][i] + intercept

            stdVec[m] = slope * stdVec[m]

        prdDf = pd.DataFrame( { 'Date' : dateList } )
        
        for m in range( nDims ):
            asset = ecoMfd.velNames[m]
            prdDf[ asset ] = prdSol[m]
            prdDf[ asset + '_std' ] = stdVec[m]

        self.prdDf = prdDf.groupby( 'Date', as_index = False ).mean()

        self.logger.info( 'Setting prediction solution took %0.2f seconds!' % \
                          ( time.time() - t0 ) )
        
        return prdDf

    def getCurAction( self, curOption, curUPrice ):

        decision  = self.getDecision( curOption, curUPrice )

        if decision == 'sell_now':
            prob    = 1.0
        elif decision == 'exec_now':
            prob    = 1.0
        elif decision == 'exec_maturity':
            prob = self.getProb( curOption )
        elif decision == 'no_action':
            prob = self.getProb( curOption )
        else:
            self.logger.error( 'Unknown decision %s' % decision )

        return ( decision, prob )

    def selOptions( self,
                    options,
                    cash,
                    maxPriceC,
                    maxPriceA,
                    maxSelCnt  = None,
                    optionType = None  ):

        t0      = time.time()

        options = self.filterOptions( options,
                                      maxPriceC,
                                      optionType )

        probs = []
        
        for option in options:
            
            prob = self.getProb( option )
            
            probs.append( prob )
            
            option[ 'Prob' ] = prob

        probs   = np.array( probs )
        sumInv  = np.sum( probs )
        
        if sumInv > 0:
            sumInv = 1.0 / sumInv

        probs = sumInv * probs
        
        self.logger.info( 'Selecting from a pool of %d contracts...',
                          len( options ) )
        
        spentHash = {}
        for asset in self.assetHash:
            spentHash[ asset ] = 0.0

        eligHash = {}
        for option in options:
            eligHash[ option[ 'optionSymbol' ] ] = 1

        totVal  = cash            
        selHash = Counter()
        selCnt  = 0
        
        while totVal > 0:

            if sum( eligHash.values() ) == 0:
                self.logger.info( 'No more eligible options found!' )
                break

            if maxSelCnt is not None:
                if selCnt >= maxSelCnt:
                    self.logger.info( 'Reached the maximum selection '
                                      'count of %d!',
                                      maxSelCnt )
                    break
            
            option = np.random.choice( options,
                                       replace = True,
                                       p       = probs )
            
            symbol   = option[ 'optionSymbol' ]
            asset    = option[ 'assetSymbol' ]
            strike   = option[ 'strike' ]
            exprDate = option[ 'expiration' ]
            oType    = option[ 'type' ]                
            oCnt     = option[ 'contractCnt' ]
            uPrice   = option[ 'unitPrice' ]
            oPrice   = uPrice * oCnt                
            cost     = oPrice + self.tradeFee
            prob     = option[ 'Prob' ]
            
            if totVal < cost:
                eligHash[ symbol ] = 0
                continue

            tmpVal = spentHash[ asset ] + oPrice

            if tmpVal > maxPriceA:
                eligHash[ symbol ] = 0
                continue
                
            if prob < self.minProb:
                eligHash[ symbol ] = 0                
                continue
            
            selHash[ symbol ]  += 1                
            spentHash[ asset ] += oPrice                
            totVal             -= cost
            selCnt             += 1

            self.logger.info( 'Selecting %s; cost is %0.2f; '
                              'success probability is %0.3f; remaining cash is %0.2f',
                              symbol,
                              cost,
                              prob,
                              totVal )
                
        self.logger.info( 'Selected a total of %d options! '
                          'Total cost of selected options: %0.2f!',
                          len( selHash ),
                          ( cash - totVal ) )
        
        self.logger.info( 'Selecting options took %0.2f seconds!',
                          ( time.time() - t0 ) )

        return selHash

    def getDecision( self, option, curUPrice ):

        decisions = [ 'exec_maturity', 'exec_now', 'sell_now' ]
        exprDate  = pd.to_datetime( option[ 'expiration' ] )
        retHash   = {}
        
        for decision in decisions:
            
            ret = self.getExpReturn( option, curUPrice, mode =  decision )
            
            retHash[ decision ] = ret

        maxRet = max( retHash.values() )

        if maxRet < 0:
            decision = 'no_action'
        elif retHash[ 'exec_now' ] == maxRet:
            decision = 'exec_now'
        elif retHash[ 'sell_now' ] == maxRet:
            decision = 'sell_now'
        elif retHash[ 'exec_maturity' ] == maxRet and \
             exprDate == self.curDate:
            decision = 'exec_now'
        else:
            decision = 'exec_maturity'

        return decision
        
    def getExpReturn( self,
                      option,
                      curUPrice,
                      mode = 'exec_maturity' ):

        validFlag = self.validateOption( option )

        if not validFlag:
            return None

        asset    = option[ 'assetSymbol' ]
        strike   = option[ 'strike' ]
        exprDate = option[ 'expiration' ]
        exprDate = pd.to_datetime( exprDate )
        uPrice   = option[ 'unitPrice' ]
        oType    = option[ 'type' ]        
        oCnt     = option[ 'contractCnt' ]

        prdDf    = self.prdDf        
        dateStr  = exprDate.strftime( '%Y-%m-%d' )
        prdHash  = dict( zip( prdDf.Date, prdDf[ asset ] ) )
        prdPrice = prdHash[ dateStr ]

        curPrice = self.assetHash[ asset ]

        fee      = self.tradeFee / oCnt

        nDays    = ( exprDate - self.curDate ).days
        tmpVal   = ( 1.0 + self.rfiDaily )**nDays
        uCost    = tmpVal * ( uPrice + fee )

        if oType == 'call':
            etaVal  = strike + uCost
        elif oType == 'put':
            etaVal  = strike - uCost
        else:
            return None

        if mode == 'exec_maturity':
            if oType == 'call':
                val = prdPrice - etaVal
            elif oType == 'put':
                val = etaVal - prdPrice
            else:
                return None
        elif mode == 'exec_now':
            if oType == 'call':
                val = curPrice - etaVal
            elif oType == 'put':
                val = etaVal - curPrice
            else:
                return None
        elif mode == 'sell_now':
            val = curUPrice - uCost
        else:
            self.logger.error( 'Unknow mode %s!', mode )
            return None

        return val

    def getNormProb( self, option ):

        oCnt   = option[ 'contractCnt' ]
        uPrice = option[ 'unitPrice' ]
        oPrice = uPrice * oCnt                
        cost   = oPrice + self.tradeFee
        prob   = self.getProb( option )

        fct = 0.0
        if cost > 0:
            fct = 1.0 / np.log( cost )

        return prob * fct
    
    def getProb( self, option ):

        validFlag = self.validateOption( option )

        if not validFlag:
            return 0.0
        
        asset    = option[ 'assetSymbol' ]
        strike   = float( option[ 'strike' ] )
        exprDate = option[ 'expiration' ]
        uPrice   = float( option[ 'unitPrice' ] )
        oType    = option[ 'type' ]
        oCnt     = option[ 'contractCnt' ]

        exprDate = pd.to_datetime( exprDate )
        
        while True:
            if exprDate.isoweekday() not in [6, 7]:
                break
            else:
                exprDate -= datetime.timedelta( days = 1 )
        
        assert oCnt > 0, 'contractCnt should be > 0!'

        fee      = self.tradeFee / oCnt

        prdDf    = self.prdDf
        dateStr  = exprDate.strftime( '%Y-%m-%d' )
        prdHash  = dict( zip( prdDf.Date, prdDf[ asset ] ) )
        prdPrice = prdHash[ dateStr ]
        stdHash  = dict( zip( prdDf.Date, prdDf[ asset + '_std' ] ) )
        stdVal   = stdHash[ dateStr ]

        stdInv = stdVal * np.sqrt(2)
        if stdInv > 0:
            stdInv = 1.0 / stdInv 

        nDays    = ( exprDate - self.curDate ).days
        tmpVal   = ( 1.0 + self.rfiDaily )**nDays

        if oType == 'call':
            etaVal  = strike + tmpVal * ( uPrice + fee )
            tmpVal1 = stdInv * ( prdPrice - etaVal ) 
            prob    = 0.5 * ( 1.0 + erf( tmpVal1 ) )
        elif oType == 'put':
            etaVal  = strike - tmpVal * ( uPrice + fee )
            tmpVal1 = stdInv * ( etaVal - prdPrice ) 
            tmpVal2 = stdInv * prdPrice
            prob    = 0.5 * ( erf( tmpVal1 ) + erf( tmpVal2 ) )
        else:
            assert False, 'Only call/put options are accepted!'

        return prob

    def filterOptions( self, options, maxPriceC, optionType = None ):
        
        subSet = []

        for option in options:
            
            asset    = option[ 'assetSymbol' ]
            exprDate = option[ 'expiration' ]
            oType    = option[ 'type' ]                    
            oCnt     = option[ 'contractCnt' ]
            uPrice   = option[ 'unitPrice' ]
            oPrice   = uPrice * oCnt                
            exprDate = pd.to_datetime( exprDate )

            if asset not in self.assetHash.keys():
                continue

            if asset not in self.ecoMfd.velNames:
                continue

            if exprDate <= self.curDate:
                continue

            if exprDate < self.minDate:
                continue
            
            if exprDate > self.maxDate:
                continue

            if oPrice > maxPriceC:
                continue

            if optionType is not None:
                if oType != optionType:
                    continue
                
            prob = self.getProb( option )

            if prob < self.minProb:
                continue
            
            subSet.append( option )

        return subSet
    
    def validateOption( self, option ):

        oCnt      = option[ 'contractCnt' ]
        asset     = option[ 'assetSymbol' ]
        exprDate  = pd.to_datetime( option[ 'expiration' ] )

        if oCnt <= 1:
            self.logger.error( 'Contract size should be >= 1!' )
            return False
        
        if asset not in self.assetHash.keys():
            self.logger.error( 'Asset %s not found in assetHash!', asset )
            return False
            
        if asset not in self.ecoMfd.velNames:
            self.logger.error( 'Contract %s: asset %s not found in the model!',
                               option[ 'optionSymbol' ], asset )
            return False
                
        if exprDate < self.curDate:
            msgStr = 'Contract %s: ' +\
                'expiration %s should be >= curDate %s'
            self.logger.error( msgStr,
                               option[ 'optionSymbol' ],
                               str( exprDate ),
                               str( self.curDate )   )
            return False

        if exprDate > self.maxDate:
            msgStr = 'Contract %s: ' +\
                'expiration %s should be <= maxDate %s'
            self.logger.error( msgStr,
                               option[ 'optionSymbol' ],
                               str( exprDate ),
                               str( self.maxDate )   )
            return False

        return True

