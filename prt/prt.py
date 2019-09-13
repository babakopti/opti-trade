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

GEO_TOL = 1.0e-2
OPT_TOL = 1.0e-9

# ***********************************************************************
# Class MfdPrt: Model object for a manifold based portfolio
# ***********************************************************************

class MfdPrt:

    def __init__(   self,
                    modFile,
                    curDate,
                    endDate,
                    assets,
                    quoteHash,
                    totAssetVal, 
                    tradeFee     = 0.0,
                    minGainRate  = 0.0,
                    strategy     = 'mad_con_mfd',
                    minProbLong  = 0.51,
                    minProbShort = 0.51,
                    verbose      = 1          ):

        self.mfdMod    = dill.load( open( modFile, 'rb' ) ) 
        self.ecoMfd    = self.mfdMod.ecoMfd
        self.curDate   = pd.to_datetime( curDate )
        self.endDate   = pd.to_datetime( endDate )

        assert self.curDate > pd.to_datetime( self.ecoMfd.minTrnDate ),\
            'minTrnDate should be before curDate!'

        assert self.endDate > self.curDate,\
            'endDate should be larger than curDate!'

        self.assets    = []
        for asset in assets:
            if asset not in self.ecoMfd.velNames:
                print( 'Dropping', asset, '; not found in the model', modFile )
                continue
            if asset not in quoteHash:
                print( 'Dropping', asset, '; not found in quoteHash!' )
                continue

            self.assets.append( asset )

        for asset in quoteHash:
            assert quoteHash[ asset ] > 0, 'Price should be positive!'

        self.quoteHash = quoteHash

        self.totAssetVal = totAssetVal
        self.tradeFee    = tradeFee
        self.minGainRate = minGainRate

        assert strategy in [ 'mad_con_mfd', 'gain_mfd' ], 'Strategy %s is not known!' % strategy

        self.strategy  = strategy

        assert minProbLong > 0,  'minProbLong should be > 0!'
        assert minProbLong < 1,  'minProbLong should be < 1!'
        assert minProbShort > 0, 'minProbShort should be > 0!'
        assert minProbShort < 1, 'minProbShort should be < 1!'

        self.minProbLong  = minProbLong
        self.minProbShort = minProbShort
        self.verbose      = verbose
        self.retDf        = None
        self.prdSol       = None
        self.stdVec       = None
        self.optFuncVals  = None

        self.setRetDf()
        self.setPrdSol()
        self.setPrdStd()

    def setRetDf( self ):

        self.retDf = pd.DataFrame()
        ecoMfd     = self.ecoMfd
        actSol     = ecoMfd.actSol
        nTimes     = ecoMfd.nTimes

        for m in range( ecoMfd.nDims ):
            asset     = ecoMfd.velNames[m]
            
            if asset not in self.assets:
                continue

            tmp       = ecoMfd.deNormHash[ asset ]
            slope     = tmp[0]
            intercept = tmp[1]
            df        = pd.DataFrame( { asset : slope * actSol[m][:nTimes] +\
                                            intercept } )

            self.retDf[ asset ] = np.log( df[ asset ] ).pct_change().dropna()

    def setPrdSol( self ):
        
        ecoMfd   = self.ecoMfd
        Gamma    = ecoMfd.getGammaArray( ecoMfd.GammaVec )
        bcVec    = ecoMfd.endSol
        nDays    = ( self.endDate - self.curDate ).days
        nMinutes = int( nDays * 8 * 60 )

        odeObj   = OdeGeoConst( Gamma    = Gamma,
                                bcVec    = bcVec,
                                bcTime   = 0.0,
                                timeInc  = 1.0,
                                nSteps   = nMinutes - 1,
                                intgType = 'LSODA',
                                tol      = GEO_TOL,
                                verbose  = self.verbose       )

        sFlag    = odeObj.solve()

        if not sFlag:
            if self.verbose > 0:
                print( 'Geodesic equation did not converge!' )
            return None

        prdSol   = odeObj.getSol()

        for m in range( ecoMfd.nDims ):
            asset     = ecoMfd.velNames[m]
            tmp       = ecoMfd.deNormHash[ asset ]
            slope     = tmp[0]
            intercept = tmp[1]
            
            for i in range( nMinutes ):
                prdSol[m][i] = slope * prdSol[m][i] + intercept

        assert prdSol.shape[0] == ecoMfd.nDims, 'Inconsistent prdSol size!'
        assert prdSol.shape[1] > 0, 'Number of minutes should be positive!'
        
        self.prdSol = prdSol

        return prdSol

    def setPrdStd( self ):

        ecoMfd = self.ecoMfd
        stdVec = ecoMfd.getConstStdVec()

        for m in range( ecoMfd.nDims ):
            asset     = ecoMfd.velNames[m]
            tmp       = ecoMfd.deNormHash[ asset ]
            slope     = tmp[0]
            stdVec[m] = slope * stdVec[m]

        self.stdVec = stdVec

        return stdVec

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
        trendHash    = self.getPrdTrends()

        self.optFuncVals = []
        
        optFunc   = self.getOptFunc()
        optCons   = self.getOptCons()
        guess     = self.getInitGuess()

        results   = minimize( fun         = optFunc, 
                              x0          = guess, 
                              method      = 'SLSQP',
                              tol         = OPT_TOL,
                              constraints = optCons        )

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
            
            assert curPrice > 0, 'Price should be positive!'

            qty      = int( weights[i] * totAssetVal / curPrice )

            totVal  += abs( qty ) * curPrice
 
            prtHash[ asset ] = weights[i]

        if self.verbose > 0:
            print( 'Building portfolio took', 
                   round( time.time() - t0, 2 ), 
                   'seconds!' )

            print( 'Total value of new portfolio:', totVal )

            print( 'Sum of wts:', sum( abs( weights ) ) )

            print( prtHash )

        return prtHash

    def getOptFunc( self ):

        strategy = self.strategy

        if strategy == 'mad_con_mfd':
            optFunc = self.getMadFunc
        elif strategy == 'gain_mfd':
            optFunc = self.getFuncGain
        else:
            assert False, 'Strategy %s is not known!' % strategy

        return optFunc

    def getOptCons( self ):

        strategy     = self.strategy
        minProbLong  = self.minProbLong 
        minProbShort = self.minProbShort
        assets       = self.assets
        nAssets      = len( assets )
        trendHash    = self.getPrdTrends()
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

            if strategy == 'mad_con_mfd':
                cons.append( { 'type' : 'ineq', 'fun' : trendFunc } )

            if False and strategy == 'mad_con_mfd':
                gainFunc  = lambda wts : self.wtsGainCheck(  wts, i ) 
                cons.append( { 'type' : 'ineq', 'fun' : gainFunc  } )

        return cons

    def getInitGuess( self ):

        minProbLong  = self.minProbLong 
        minProbShort = self.minProbShort
        assets       = self.assets
        nAssets      = len( assets )
        trendHash    = self.getPrdTrends()
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
                continue
 
        return guess
        
    def getPrdTrends( self ):

        quoteHash = self.quoteHash
        ecoMfd    = self.ecoMfd
        nDims     = ecoMfd.nDims
        prdSol    = self.prdSol
        stdVec    = self.stdVec
        nMinutes  = prdSol.shape[1]

        assert nMinutes > 0, 'Number of minutes should be positive!'

        trendHash = {}
        
        for m in range( nDims ):
            asset    = ecoMfd.velNames[m]

            if asset not in self.assets:
                continue

            curPrice = quoteHash[ asset ]
            trend    = 0.0
            prob     = 0.0

            for i in range( nMinutes ):

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
            
            trend /= nMinutes
            prob  /= nMinutes
            
            trendHash[ asset ] = ( trend, prob )
            
        return ( trendHash )

    def wtsRatioCheck( self, wts, assetId ):

        asset     = self.assets[assetId]
        curPrice  = self.quoteHash[ asset ]

        assert curPrice > 0, 'Price should be positive!'
        
        val  = abs( wts[assetId] ) * self.totAssetVal / curPrice 
        val  = ( int( val ) - val ) * curPrice / self.totAssetVal

        return val

    def wtsGainCheck( self, wts, assetId ):

        ecoMfd      = self.ecoMfd        
        quoteHash   = self.quoteHash
        totAssetVal = self.totAssetVal 
        tradeFee    = self.tradeFee
        minGainRate = self.minGainRate
        assets      = self.assets
        asset       = assets[ assetId ]
        prdSol      = self.prdSol
        nAssets     = len( assets )
        nMinutes    = prdSol.shape[1]

        assert nMinutes > 0, 'Number of minutes should be positive!'

        curPrice = quoteHash[ asset ]

        assert curPrice > 0, 'Price should be positive!'
            
        for m in range( ecoMfd.nDims ):
            if ecoMfd.velNames[m] == asset:
                break

        assert m < ecoMfd.nDims, 'Asset %s not found in the model!' % asset

        prdPrice = np.mean( prdSol[m] )
        qty      = wts[assetId] * totAssetVal / curPrice
        gain     = qty * ( prdPrice - curPrice ) - tradeFee
        rate     = gain / ( qty * curPrice )
            
        return rate - minGainRate

    def getMadFunc( self, wts ):

        mad = ( self.retDf - self.retDf.mean() ).dot( wts ).abs().mean()

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
                if ecoMfd.velNames[m] == asset:
                    break

            assert m < ecoMfd.nDims, 'Asset %s not found in the model!' % asset

            prdPrice = np.mean( prdSol[m] ) 
            qty      = int( wts[assetId] * totAssetVal / curPrice )
            gain    += qty * ( prdPrice - curPrice ) - tradeFee

        val = -gain

        self.optFuncVals.append( val )

        return val

    def checkCons( self, cons, wts ):

        for con in cons:
            conFunc = con[ 'fun' ]
            
            if con[ 'type' ] == 'eq':
                assert abs( conFunc( wts ) ) < OPT_TOL, \
                    'Equality constraint not satisfied!'
            elif con[ 'type' ] == 'ineq':
                assert conFunc( wts ) >= -OPT_TOL, \
                    'Inequality constraint not satisfied!'
            else:
                assert False, 'Unknown constraint type!'

    def pltItrers( self ):
        plt.plot( self.optFuncVals )

