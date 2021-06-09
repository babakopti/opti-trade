import os, sys

sys.path.append("..")

from mfd.ecoMfdNN import EcoMfdNN

ecoMfdNN = EcoMfdNN(
    varNames=["y1_cum", "y2_cum", "y3_cum"],
    velNames=["y1", "y2", "y3"],
    dateName="Date", 
    dfFile="../../geo-paper-1st-order/analysis/random_signals.pkl",
    minTrnDate="2004-01-15", 
    maxTrnDate="2004-02-15", 
    maxOosDate="2004-02-25",
    hLayerSizes=[5,10],                    
    trmFuncDict  = {},
    optType      = 'BFGS',
    maxOptItrs   = 5000, 
    optGTol      = 1.0e-10,
    optFTol      = 1.0e-10,
    stepSize     = 1000.0,
    factor       = 4.0e-5,
    regCoef      = 1.0e-10,
    regL1Wt      = 0.0,
    nPca         = None,
    diagFlag     = True,
    srelFlag     = True,                                        
    endBcFlag    = True,
    varCoefs     = None,
    srcCoefs     = None,
    srcTerm      = None,
    atnFct       = 1.0,
    mode         = 'day',
    logFileName  = None,
    verbose      = 1
)

print(ecoMfdNN.nTimes, ecoMfdNN.nDims, ecoMfdNN.nParams, ecoMfdNN.nGammaVec)

sFlag = ecoMfdNN.setParams()

#print(ecoMfdNN.params)
#print(ecoMfdNN.getGamma(ecoMfdNN.params))
#print(ecoMfdNN.cumSol[0][1], ecoMfdNN.cumSol[-1][1])
#assert sFlag

ecoMfdNN.pltResults()

