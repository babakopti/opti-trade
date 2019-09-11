# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os

sys.path.append( os.path.abspath( '../' ) )

from mod.mfdMod import MfdMod
from ode.odeGeo import OdeGeoConst 

import dill
import numpy as np
import pandas as pd
import scipy as sp

from collections import defaultdict
from scipy.special import erf

# ***********************************************************************
# Some parameters
# ***********************************************************************

NULL  = 0
LONG  = 1
SHORT = 2

GEO_TOL = 1.0e-2

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
                    minProbLong  = 0.75,
                    minProbShort = 0.75,
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

        self.quoteHash = quoteHash

        self.totAssetVal = totAssetVal
        self.tradeFee    = tradeFee
        self.minGainRate = minGainRate

        assert strategy == 'mad_con_mfd', 'Only mad_con_mfd is currently supported!'

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
        nDays    = ( ecoMfd.endDate - ecoMfd.curDate ).days
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
        stdVec = self.getConstStdVec()

        for m in range( ecoMfd.nDims ):
            asset     = ecoMfd.velNames[m]
            tmp       = ecoMfd.deNormHash[ asset ]
            slope     = tmp[0]
            stdVec[m] = slope * stdVec[m]

        self.stdVec = stdVec

        return stdVec
        
    def getPortfolio( self ):

        ecoMfd       = self.ecoMfd
        assets       = self.assets
        quoteHash    = self.quoteHash
        minProbLong  = self.minProbLong 
        minProbShort = self.minProbShort
        totAssetVal  = self.totAssetVal 
        tradeFee     = self.tradeFee
        nAssets      = len( assets )
        trendHash    = self.getPrdTrends()
        guess        = np.ones( nAssets )

        cons         = [ { 'type' : 'eq', 'fun' : self.wtsSumCheck } ] 

        for i in range( nAssets ):
            asset = assets[i]
            trend = trendHash[ asset ][0]
            prob  = trendHash[ asset ][1]

            if trend > 0 and prob >= minProbLong:
                trendFunc = lambda wts : wts[i]
                gainFunc  = lambda wts : self.wtsGainCheck( wts, asset = asset ) 
            elif trend < 0 and prob >= minProbShort:
                trendFunc = lambda wts : -wts[i]
                gainFunc  = lambda wts : self.wtsGainCheck( wts, asset = asset ) 
            else:
                continue

            cons.append( { 'type' : 'ineq', 'fun' : trendFunc } )
            cons.append( { 'type' : 'ineq', 'fun' : gainFunc  } )

        results   = minimize( fun         = self.getMad, 
                              x0          = guess, 
                              method      = 'COBYLA',
                              constraints = cons      )
    
        weights   = results.x

        assert len( weights ) == nAssest,\
            'Inconsistent size of weights!'

        prtHash = {}
        for i in range( nAssets ):
            asset    = assets[i]
            curPrice = quoteHash[ asset ]
            
            assert curPrice > 0, 'Price should be positive!'

            prtHash[ assets ] = int( weights[i] * totAssetVal / curPrice )

        return prtHash
       
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

    def wtsGainCheck( self, wts, asset ):

        ecoMfd      = self.ecoMfd        
        quoteHash   = self.quoteHash
        totAssetVal = self.totAssetVal 
        tradeFee    = self.tradeFee
        minGainRate = self.minGainRate
        prdSol      = self.prdSol
        nMinutes    = prdSol.shape[1]

        assert nMinutes > 0, 'Number of minutes should be positive!'

        curPrice = quoteHash[ asset ]

        assert curPrice > 0, 'Price should be positive!'
            
        for m in range( ecoMfd.nDims ):
            if ecoMfd.velNames[m] == asset:
                break

        assert m < ecoMfd.nDims, 'Asset %s not found in the model!' % asset

        prdPrice = 0.0
        for i in range( nMinutes ):
            prdPrice += prdSol[m][i]

        prdPrice /= nMinutes

        qty  = wts[m] * totAssetVal / curPrice
        gain = qty * ( prdPrice - curPrice ) - tradeFee
        rate = gain / curPrice
         
        if rate >= minGainRate:
            fct = 1.0
        else:
            fct = -1.0
            
        return fct

    def wtsSumCheck( self, wts ):
        return sum( abs( wts ) ) - 1

    def getMad( self, wts ):
        return ( self.retDf - self.retDf.mean() ).dot( wts ).abs().mean()
