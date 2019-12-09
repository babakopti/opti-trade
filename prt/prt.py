# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os

sys.path.append( os.path.abspath( '../' ) )

from mod.mfdMod import MfdMod
from ode.odeGeo import OdeGeoConst 

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

# ***********************************************************************
# Some parameters
# ***********************************************************************

ODE_TOL   = 1.0e-2
OPT_TOL   = 1.0e-8
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
                    totAssetVal, 
                    tradeFee     = 0.0,
                    strategy     = 'mad',
                    minProbLong  = 0.5,
                    minProbShort = 0.5,
                    vType        = 'vel',
                    fallBack     = 'macd',
                    optTol       = OPT_TOL,
                    verbose      = 1          ):

        self.mfdMod    = dill.load( open( modFile, 'rb' ) ) 
        self.ecoMfd    = self.mfdMod.ecoMfd
        
        if vType == 'vel':
            self.vList = self.ecoMfd.velNames
        elif vType == 'var':
            self.vList = self.ecoMfd.varNames
        else:
            assert False, 'Unknown vType %s' % vType

        self.vType = vType

        self.assets    = []
        for asset in assets:
            if asset not in self.vList:
                print( 'Dropping', asset, '; not found in the model', modFile )
                continue

            self.assets.append( asset )

        self.nRetTimes   = nRetTimes
        self.nPrdTimes   = nPrdTimes
        self.totAssetVal = totAssetVal
        self.tradeFee    = tradeFee

        assert strategy in [ 'mad', 'gain', 'gain_mad', 'prob' ], \
            'Strategy %s is not known!' % strategy

        self.strategy  = strategy

        assert minProbLong > 0,  'minProbLong should be > 0!'
        assert minProbLong < 1,  'minProbLong should be < 1!'
        assert minProbShort > 0, 'minProbShort should be > 0!'
        assert minProbShort < 1, 'minProbShort should be < 1!'

        self.minProbLong  = minProbLong
        self.minProbShort = minProbShort
        self.fallBack     = fallBack
        self.optTol       = optTol
        self.verbose      = verbose
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
            if self.verbose > 0:
                print( 'Geodesic equation did not converge!' )
            return None

        prdSol   = odeObj.getSol()

        for m in range( ecoMfd.nDims ):
            asset     = self.vList[m]
            tmp       = ecoMfd.deNormHash[ asset ]
            slope     = tmp[0]
            intercept = tmp[1]
            
            for i in range( nPrdTimes ):
                prdSol[m][i] = slope * prdSol[m][i] + intercept

        assert prdSol.shape[0] == ecoMfd.nDims, 'Inconsistent prdSol size!'
        assert prdSol.shape[1] > 0, 'Number of minutes should be positive!'
        
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
        ecoMfd    = self.ecoMfd
        nDims     = ecoMfd.nDims
        prdSol    = self.prdSol
        stdVec    = self.stdVec
        nPrdTimes = prdSol.shape[1]
        perfs     = ecoMfd.getOosTrendPerfs( self.vType )

        assert nPrdTimes > 0, 'nPrdTimes should be positive!'

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
                if perfs[m]:
                    self.trendHash[ asset ] = ( trend, prob )
                else:
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

        t0           = time.time()
        strategy     = self.strategy
        ecoMfd       = self.ecoMfd
        assets       = self.assets
        quoteHash    = self.quoteHash
        minProbLong  = self.minProbLong 
        minProbShort = self.minProbShort
        totAssetVal  = self.totAssetVal 
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

        if self.verbose > 0:
            print( results[ 'message' ] )
            print( 'Optimization success:', results[ 'success' ] )
            print( 'Number of function evals:', results[ 'nfev' ] )

        weights   = results.x

        assert len( weights ) == nAssets,\
            'Inconsistent size of weights!'

        self.checkCons( optCons, weights )                   
            
        prtHash = {}
        totVal  = 0.0
        for i in range( nAssets ):
            asset    = assets[i]
            curPrice = quoteHash[ asset ]
            
            assert curPrice > 0, 'Price for %s should be positive instead found %s!' \
                % ( asset, curPrice )

            qty      = int( weights[i] * totAssetVal / curPrice )

            totVal  += abs( qty ) * curPrice
 
            prtHash[ asset ] = weights[i]

        if self.verbose > 0:
            print( 'Building portfolio took', 
                   round( time.time() - t0, 2 ), 
                   'seconds!' )

            print( 'Total value of new portfolio:', totVal )

            print( 'Sum of wts:', sum( abs( weights ) ) )

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
        tradeFee    = self.tradeFee
        assets      = self.assets
        prdSol      = self.prdSol
        nAssets     = len( assets )
        gain        = 0.0

        for assetId in range( nAssets ):

            asset    = assets[assetId]
            curPrice = quoteHash[ asset ]

            assert curPrice > 0, 'Price should be positive!'
            
            for m in range( ecoMfd.nDims ):
                if self.vList[m] == asset:
                    break

            assert m < ecoMfd.nDims, 'Asset %s not found in the model!' % asset

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

        assert val >= 0.0, 'Invalid value %f of probability!' % val
        assert val <= 1.0, 'Invalid value %f of probability!' % val

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

        assert m < nDims, 'm should be smaller than nDims!'

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
                assert abs( conFunc( wts ) ) < self.optTol, \
                    'Equality constraint not satisfied!'
            elif con[ 'type' ] == 'ineq':
                assert conFunc( wts ) >= -self.optTol, \
                    'Inequality constraint not satisfied!'
            else:
                assert False, 'Unknown constraint type!'

        val = np.abs( np.sum( np.abs( wts ) ) - 1.0 )

        assert val < self.optTol, \
            'The weights dp not sum up to 1.0!' 

        for i in range( nAssets ):
            asset = assets[i]
            wt    = wts[i]
            trend = trendHash[ asset ][0]
            val   = trend * wt

            assert val >= -self.optTol, \
                'The weight %0.4f for asset %s does not match predicted trend!' \
                % ( wt, asset )

        print( 'All constraints are satisfied!' )

    def pltIters( self ):
        plt.plot( self.optFuncVals, '-o' )
        plt.show()

