# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import datetime
import dill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


sys.path.append( os.path.abspath( '../..' ) )

import utl.utils as utl
from mod.mfdMod import MfdMod

# ***********************************************************************
# Set some parameters and read data
# ***********************************************************************

modFileName = 'model.dill'
resFileName = 'res_df.pkl'

srcFct      = 1.0
dfFile      = '../data/dfFile_2016plus.pkl'

minTrnDate  = pd.to_datetime( '2017-02-01 09:00:00' )
maxTrnDate  = pd.to_datetime( '2018-01-31 09:00:00' )
maxOosDate  = pd.to_datetime( '2018-02-10 23:59:00' )

indices     = [ 'INDU', 'NDX', 'SPX', 'RUT', 'OEX',  
                'MID',  'SOX', 'RUI', 'RUA', 'TRAN',
                'HGX',  'TYX', 'XAU'               ] 

futures     = [ 'ES', 'NQ', 'US', 'YM', 'RTY', 'EMD', 'QM' ]

ETFs        = [ 'QQQ', 'SPY', 'DIA', 'MDY', 'IWM', 'OIH', 
                'SMH', 'XLE', 'XLF', 'XLU', 'EWJ'          ]

velNames    = ETFs + indices + futures

# ***********************************************************************
# Some utilities
# ***********************************************************************

def buildMod( srcTerm ):
    
    mfdMod = MfdMod(    dfFile       = dfFile,
                        minTrnDate   = minTrnDate,
                        maxTrnDate   = maxTrnDate,
                        maxOosDate   = maxOosDate,
                        velNames     = velNames,
                        optType      = 'SLSQP',
                        maxOptItrs   = 200,
                        optGTol      = 1.0e-2,
                        optFTol      = 1.0e-2,
                        factor       = 4.0e-5,
                        regCoef      = 1.0e-3,
                        smoothCount  = None,
                        srcTerm      = srcTerm,
                        logFileName  = None,
                        verbose      = 1        )
    
    validFlag = mfdMod.build()

    print( 'Success :', validFlag )

    return mfdMod

def getGradient( vec ):

    q1 = np.gradient( vec )
    q2 = np.gradient( q1 )
    q3 = np.gradient( q2 )    
    q4 = np.gradient( q3 )
    q5 = np.gradient( q4 )

    ret = q1 + (1.0 / 2) * q2 #+ ( 1.0 / 6 ) * q3 + ( 1.0 / 24 ) * q4 + ( 1.0 / 120 ) * q5

    return ret
    
def getSrcTerm( mfdMod ):

    ecoMfd   = mfdMod.ecoMfd

    ecoMfd.logFileName = None
    ecoMfd.logger = utl.getLogger( None, 1 )

    nDims    = ecoMfd.nDims
    nTimes   = ecoMfd.nTimes
    nSteps   = ecoMfd.nSteps
    sol      = ecoMfd.getSol( ecoMfd.GammaVec ).getSol()
    actSol   = ecoMfd.actSol
    Gamma    = ecoMfd.getGammaArray( ecoMfd.GammaVec )

    src      = np.zeros( shape = ( nDims, nTimes ), dtype = 'd' )
    tmpVec1  = np.zeros( shape = ( nDims, nDims, nTimes ), dtype = 'd' )
    tmpVec2  = np.zeros( shape = ( nDims, nDims, nTimes ), dtype = 'd' )    

    for a in range( nDims ):
        for b in range( nDims ):
            tmpVec1[a][b] = 2.0 * ( sol[a] - actSol[a] ) * actSol[b]
            tmpVec2[a][b] = ( sol[a] - actSol[a] ) * ( sol[b] - actSol[b] ) 
            
    for m in range( nDims ):
        src[m] = np.gradient( sol[m], edge_order = 2 ) - \
                 np.gradient( actSol[m], edge_order = 2 )
        
    src += np.tensordot( Gamma, tmpVec1, ( ( 1, 2 ), ( 0, 1 ) ) ) +\
           np.tensordot( Gamma, tmpVec2, ( ( 1, 2 ), ( 0, 1 ) ) )
                                   
    src = -src
        
    return src

# ***********************************************************************
# Build original model
# ***********************************************************************

mfdMod = buildMod( None )
#mfdMod.save( modFileName )    
#mfdMod = dill.load( open( modFileName, 'rb' ) )
ecoMfd = mfdMod.ecoMfd
nDims  = ecoMfd.nDims
nTimes = ecoMfd.nTimes
nSteps = ecoMfd.nSteps
actSol = ecoMfd.actSol

# ***********************************************************************
# Get raw sol
# ***********************************************************************

print( 'Getting the raw solution...' )

rawSol = ecoMfd.getSol( ecoMfd.GammaVec ).getSol()

# ***********************************************************************
# Plugin actual res
# ***********************************************************************

print( 'Getting the solution with exact source term...' )

print( 'Getting source term...' )

excSrcTerm = getSrcTerm( mfdMod )

ecoMfd.srcTerm = srcFct * excSrcTerm

print( 'Solving ODE...' )

excSrcSol = ecoMfd.getSol( ecoMfd.GammaVec ).getSol()

# ***********************************************************************
# Reconstruct res and Plugin
# ***********************************************************************

# ecoMfd   = mfdMod.ecoMfd
# nDims    = ecoMfd.nDims
# nTimes   = ecoMfd.nTimes
# nSteps   = ecoMfd.nSteps
# actSol   = ecoMfd.actSol

# print( 'Getting the solution with reconstructed source term...' )

# print( 'Forming source term...' )

# recSrcTerm = np.zeros( shape = ( nDims, nTimes ), dtype = 'd' )

# for m in range( nDims ):
#     excVals    = excSrcTerm[m]
#     hist, bins = np.histogram( excVals, bins = nTimes )
#     midpoints  = bins[:-1] + np.diff( bins ) / 2
#     cdf        = np.cumsum( hist )
#     cdf        = cdf / cdf[-1]
#     vals       = np.random.rand( nTimes )
#     binInds    = np.searchsorted( cdf, vals )
#     recVals    = midpoints[binInds]
    
#     recSrcTerm[m] = recVals

# print( 'Solving the ODE...' )

# recSrcSol = ecoMfd.getSol( ecoMfd.GammaVec ).getSol()

# ***********************************************************************
# Plot
# ***********************************************************************

for m in range( nDims ):
    
    plt.plot( actSol[m] )
    plt.plot( rawSol[m] )
    plt.plot( excSrcSol[m] )
#    plt.plot( recSrcSol[m] )

    legends = [ 'Actual',
                'Raw',
                'Exact Term']#,
#                'Recon. Term' ]
    plt.legend( legends )
    plt.ylabel( velNames[m] )
    plt.show()
