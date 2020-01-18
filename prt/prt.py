# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import dill
import time
import talib
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

from collections import defaultdict
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
# Class MfdPrt: Model object for a manifold based portfolio
# ***********************************************************************

class MfdPrt:

    def __init__(   self,
                    modFile,
                    assets,
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

        self.assets    = []
        for asset in assets:
            if asset not in self.vList:
                self.logger.warning( 'Dropping %s ; not found in the model %s',
                                     asset,
                                     modFile )
                continue

            self.assets.append( asset )

        self.nRetTimes   = nRetTimes
        self.nPrdTimes   = nPrdTimes

        if strategy not in [ 'mad', 'gain', 'gain_mad', 'prob', 'equal' ]:
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
        self.quoteHash    = None
        self.trendHash    = None

        self.setRetDf()
        self.setPrdSol()
        self.setPrdStd()
        self.setQuoteHash()
        self.setPrdTrends()

    def setRetDf( self ):

        self.retDf = pd.DataFrame()
        nRetTimes  = self.nRetTimes
        ecoMfd     = self.ecoMfd
        actSol     = ecoMfd.actSol
        actOosSol  = ecoMfd.actOosSol
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
            bcVec[m]  = actOosSol[m][-1] 

        odeObj   = OdeGeoConst( Gamma    = Gamma,
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

    def setQuoteHash( self ):

        ecoMfd    = self.ecoMfd
        nDims     = ecoMfd.nDims
        actOosSol = ecoMfd.actOosSol
        quoteHash = {}

        for m in range( nDims ):
            asset     = self.vList[m]
            tmp       = ecoMfd.deNormHash[ asset ]
            slope     = tmp[0]
            intercept = tmp[1]

            quoteHash[ asset ] = slope * actOosSol[m][-1] + intercept 

        self.quoteHash = quoteHash

        return quoteHash

    def setPrdTrends( self ):

        quoteHash = self.quoteHash
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

            curPrice = quoteHash[ asset ]
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
                if perfs[m]: #and mfdMod.converged:
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
        quoteHash    = self.quoteHash
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
        elif strategy == 'gain':
            optFunc = self.getGainFunc
        elif strategy == 'gain_mad':
            optFunc = self.getGainMadFunc
        elif strategy == 'prob':
            optFunc = self.getProbFunc
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

    def getGainFunc( self, wts ):

        ecoMfd      = self.ecoMfd        
        quoteHash   = self.quoteHash
        totAssetVal = self.totAssetVal 
        assets      = self.assets
        prdSol      = self.prdSol
        nAssets     = len( assets )
        gain        = 0.0

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

            prdPrice = np.mean( prdSol[m] )

            curPrice = np.log( curPrice )
            prdPrice = np.log( prdPrice )
            gain    += wts[assetId] * ( prdPrice - curPrice ) /  curPrice

        val = 1.0 / gain

        sumFunc = lambda wts : ( sum( abs( wts ) ) - 1.0 )

        if abs( sumFunc( wts ) ) < self.optTol:
            self.optFuncVals.append( val )

        return val

    def getGainMadFunc( self, wts ):

        gainInv = self.getGainFunc( wts )
        mad     = self.getMadFunc(  wts )
        val     = gainInv * mad

        sumFunc = lambda wts : ( sum( abs( wts ) ) - 1.0 )

        if abs( sumFunc( wts ) ) < self.optTol:
            self.optFuncVals.append( val )

        return val

    def getProbFunc( self, wts ):

        assets     = self.assets
        nAssets    = len( assets )
        trendHash  = self.trendHash
        val        = 0.0

        for i in range( nAssets ):
            asset = assets[i]
            prob  = trendHash[ asset ][1]
            val  += abs( wts[i] ) * prob

        tmpSum = np.sum( abs( wts ) )
        val   /= tmpSum
        val    = 1.0 - val

        if val < 0.0:
            self.logger.error( 'Invalid value %f of probability!', val )
            assert False, 'Invalid value %f of probability!' % val

        if val > 1.0:
            self.logger.error( 'Invalid value %f of probability!', val )
            assert False, 'Invalid value %f of probability!' % val

        if abs( tmpSum - 1.0 ) < self.optTol:
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

