# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os

sys.path.append( os.path.abspath( '../' ) )

from mfd.ecoMfd import EcoMfdConst as EcoMfd

import dill
import pickle as pk
import numpy as np
import pandas as pd
import scipy as sp

# ***********************************************************************
# Some definitions
# ***********************************************************************

DIAG_FLAG = True

# ***********************************************************************
# Class MfdMod: Model object that builds a manifold based model
# ***********************************************************************

class MfdMod:

    def __init__(   self,
                    dfFile,
                    minTrnDate, 
                    maxTrnDate,
                    maxOosDate,
                    velNames,
                    maxOptItrs   = 100, 
                    optGTol      = 1.0e-4,
                    optFTol      = 1.0e-8,
                    factor       = 4.0e-5,
                    regCoef      = None,
                    minMerit     = 0.0,
                    minTrend     = 0.0,
                    maxBias      = 1.0,
                    varFiltFlag  = False,
                    validFlag    = False,
                    smoothCount  = None,
                    verbose      = 1          ):


        self.dfFile      = dfFile
        self.minTrnDate  = minTrnDate
        self.maxTrnDate  = maxTrnDate
        self.maxOosDate  = maxOosDate
        self.maxOptItrs  = maxOptItrs
        self.optGTol     = optGTol
        self.optFTol     = optFTol
        self.minMerit    = minMerit
        self.minTrend    = minTrend
        self.maxBias     = maxBias
        self.varFiltFlag = varFiltFlag
        self.validFlag   = validFlag
        self.verbose     = verbose
        self.ecoMfd      = None
        
        if regCoef is None:
            self.regCoef    = 0.0
            self.optRegFlag = True
        else:
            self.regCoef    = regCoef
            self.optRegFlag = False
        
        self.factor   = factor

        self.velNames = velNames

        self.varNames = []

        for velName in self.velNames:

            tmpList = velName.split( '_' )
            
            if tmpList[-1] == 'Diff':
                varName = '_'.join( tmpList[:-1] )
            else:
                varName = velName + '_Cumul'

            self.varNames.append( varName )

        self.trmFuncDict = {}

        if smoothCount is not None :
            for velName in self.velNames:
                self.trmFuncDict[ velName ] = lambda x : x.rolling( int( smoothCount ), 
                                                                    win_type = 'blackman',
                                                                    center   = True ).mean()

    def build( self ):

        print( 'Building a manifold...' )

        sFlag  = self.setMfd()

        if self.verbose > 0:
            self.echoMod()
        
        if not sFlag:
            return False

        if self.varFiltFlag:
            dropFlag = self.filtVars()
            if dropFlag:
                sFlag = self.setMfd()

                if self.verbose > 0:
                    self.echoMod()

                if not sFlag:
                    return False

        if self.optRegFlag:
            self.regCoef = self.optReg()
            sFlag = self.setMfd()

            if self.verbose > 0:
                self.echoMod()

            if not sFlag:
                return False

        if self.validFlag:
            sFlag = self.validate()

            if not sFlag:
                return False

        ecoMfd = self.ecoMfd
        merit  = ecoMfd.getMerit()
        merit  = min( merit, ecoMfd.getOosMerit() )
        trend  = ecoMfd.getOosTrendCnt()
        nDims  = ecoMfd.nDims
        bias   = 0

        for varId in range( nDims ):
            bias = max( bias, ecoMfd.getRelBias( varId ) )
        
        if merit < self.minMerit:
            print( 'Merit does not meet criteria!' )
            sFlag = False

        if trend < self.minTrend:
            print( 'Trend does not need criteria!' )
            sFlag = False
            
        if bias > self.maxBias:
            print( 'Bias does not meet criteria' )
            sFlag = False

        if not sFlag:
            return False

        return True

    def setMfd( self ):

        self.ecoMfd = EcoMfd( varNames     = self.varNames,
                              velNames     = self.velNames,
                              dateName     = 'Date', 
                              dfFile       = self.dfFile,
                              minTrnDate   = self.minTrnDate,
                              maxTrnDate   = self.maxTrnDate,
                              maxOosDate   = self.maxOosDate,
                              trmFuncDict  = self.trmFuncDict,
                              optType      = 'GD', 
                              maxOptItrs   = self.maxOptItrs, 
                              optGTol      = self.optGTol,
                              optFTol      = self.optFTol,
                              stepSize     = 1.0,
                              factor       = self.factor,
                              regCoef      = self.regCoef,
                              regL1Wt      = 0.0,
                              nPca         = None,
                              diagFlag     = DIAG_FLAG,
                              endBcFlag    = True,
                              verbose      = self.verbose        )        

        sFlag = self.ecoMfd.setGammaVec()

        if not sFlag:
            print( 'Warning: did not converge!' )

        return sFlag

    def save( self, outModFile ):

        with open( outModFile, 'wb' ) as fHd:
            dill.dump( self, fHd, pk.HIGHEST_PROTOCOL )

    def echoMod( self ):

        ecoMfd  = self.ecoMfd
        nDims   = ecoMfd.nDims
        biasVec = []

        for m in range( nDims ):
            bias = ecoMfd.getRelBias( m )
            biasVec.append( bias )

        print( 'Manifold merit    :', ecoMfd.getMerit(),       '\n' )
        print( 'Manifold oos merit:', ecoMfd.getOosMerit(),    '\n' )
        print( 'Manifold oos velocity trend match:', ecoMfd.getOosTrendCnt( 'vel' ), '\n' )
        print( 'Manifold oos variable trend match:', ecoMfd.getOosTrendCnt( 'var' ), '\n' )
        print( 'Manifold max bias :', max( biasVec ),          '\n' )            

        print( ecoMfd.getTimeDf(), '\n' )

    def filtVars( self ):

        if not self.varFiltFlag:
            return

        ecoMfd      = self.ecoMfd
        varNames    = self.varNames
        velNames    = self.velNames
        newVarNames = []
        newVelNames = []
        dropFlag    = False
        for varId in range( ecoMfd.nDims ):

            varName = varNames[varId]
            velName = velNames[varId]
            merit   = ecoMfd.getMerit( [ varName ] )
            bias    = ecoMfd.getRelBias( varId )

            if merit < self.minMerit:
                dropFlag = True
                continue

            if bias > self.maxBias:
                dropFlag = True
                continue

            newVarNames.append( varName )
            newVelNames.append( velName )        
    
        self.varNames = newVarNames
        self.velNames = newVelNames

        return dropFlag

    def crossVal( self,
                  minYrs,
                  incYrs,
                  regCoef = 0.0   ):

        minDt     = pd.to_datetime( self.minTrnDate )
        maxDt     = pd.to_datetime( self.maxTrnDate )
        maxOosDt  = pd.to_datetime( self.maxOosDate )

        if 2 * incYrs > ( maxDt.year - minDt.year ):
            print( 'Increment is too small! No cross validation done!' )
            return None

        if 3 * incYrs > ( maxOosDt.year - minDt.year ):
            print( 'Increment is too small! No cross validation done!' )
            return None

        maxDt    = minDt + pd.DateOffset( years = int( minYrs ) )

        maxMaxDt = pd.to_datetime( self.maxOosDate ) -\
                   pd.DateOffset( years = incYrs )
        maxMaxDt = min( maxMaxDt, pd.to_datetime( self.maxTrnDate ) )

        maxTrnDates = []
        while maxDt <= maxMaxDt:
            maxTrnDates.append( maxDt.strftime( '%Y-%m-%d' ) )
            maxDt = maxDt + pd.DateOffset( years = int( incYrs ) )

        ecoMfds = []
        for maxTrnDate in maxTrnDates:

            maxOosDt   = pd.to_datetime( maxTrnDate ) +\
                         pd.DateOffset( years = incYrs )

            maxOosDate = maxOosDt.strftime( '%Y-%m-%d' )

            ecoMfd     = EcoMfd( varNames     = self.varNames,
                                 velNames     = self.velNames,
                                 dateName     = 'Date', 
                                 dfFile       = self.dfFile,
                                 minTrnDate   = self.minTrnDate,
                                 maxTrnDate   = maxTrnDate,
                                 maxOosDate   = maxOosDate,
                                 trmFuncDict  = self.trmFuncDict,
                                 optType      = 'GD', 
                                 maxOptItrs   = self.maxOptItrs, 
                                 optGTol      = self.optGTol,
                                 optFTol      = self.optFTol,
                                 stepSize     = 1.0,
                                 factor       = self.factor,
                                 regCoef      = regCoef,
                                 regL1Wt      = 0.0,
                                 nPca         = None,
                                 diagFlag     = DIAG_FLAG,
                                 endBcFlag    = True,
                                 verbose      = self.verbose        )

            ecoMfds.append( ecoMfd )

        return ecoMfds

    def optReg( self ):
        
        if not self.optRegFlag:
            return

        if self.verbose > 0:
            print( 'Running optimization for regularization coefficient!' )

        options    = {    'disp'       : False,
                          'gtol'       : 1e-05,
                          'eps'        : 1e-03,
                          'return_all' : False,
                          'maxiter'    : 200,
                          'norm'       : np.inf         }

        res  = sp.optimize.minimize( self.optRegFunc,
                                     x0      = 0.0,
                                     method  = 'CG',
                                     options = options   )
        print( res )
        print( res[ 'x' ] )

        return res[ 'x' ]

    def optRegFunc( self, regCoef ):
        
        minDt   = pd.to_datetime( self.minTrnDate )
        maxDt   = pd.to_datetime( self.maxTrnDate )
        minYrs  = int( ( maxDt.year - minDt.year ) / 2.0 )
        incYrs  = 1.0
        val     = 0.0

        ecoMfds = self.crossVal( minYrs  = minYrs,
                                 incYrs  = incYrs,
                                 regCoef = regCoef )

        for ecoMfd in ecoMfds:
            val += ecoMfd.getOosMerit()

        fct = len( ecoMfds )
        if fct > 0:
            fct = 1.0 / fct

        val *= fct

        return val

    def validate( self ):

        validFlag = True

        oosMeritAvg = self.optRegFunc( self.regCoef )
        
        if self.verbose > 0:
            print( 'Average OOS merit:', oosMeritAvg )

        validFlag = validFlag & (oosMeritAvg >= self.minMerit)
        
        return validFlag
