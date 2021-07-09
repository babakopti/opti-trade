import os, sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("..")

from scipy.integrate import trapz
import tensorflow as tf
from tensorflow import keras

from mfd.ecoMfdNN import EcoMfdNN
from mfd.ecoMfd import EcoMfdConst

ecoMfdNN = EcoMfdNN(
    varNames=["y1_cum", "y2_cum", "y3_cum"],
    velNames=["y1", "y2", "y3"],
    dateName="Date", 
    dfFile="../../geo-paper-1st-order/analysis/random_signals.pkl",
    minTrnDate="2004-01-06 09:00:00", 
    maxTrnDate="2004-03-15 16:59:00", 
    maxOosDate="2004-03-29 16:59:00",
    hLayerSizes=[5],                    
    trmFuncDict  = {},
    optType      = 'SLSQP',
    maxOptItrs   = 100, 
    optGTol      = 1.0e-20,
    optFTol      = 1.0e-20,
    stepSize     = 10000.0,
    factor       = 4.0e-2,
    regCoef      = 0.0,
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

t0 = time.time()
sFlag = ecoMfdNN.setParams()
#sFlag = ecoMfdNN.setGammaVec()
print(sFlag)
print("Runtime:", (time.time() - t0))
ecoMfdNN.pltResults()

Gamma = ecoMfdNN.getGamma(ecoMfdNN.params)
print("Diff0:", np.linalg.norm(Gamma[68]-Gamma[0]))
print("Diff1:", np.linalg.norm(Gamma[68]-Gamma[66]))
print(Gamma.shape)
print(ecoMfdNN.params.shape)
y = np.zeros(shape=(ecoMfdNN.nTimes))
for tsId in range(ecoMfdNN.nTimes):
    y[tsId] = np.linalg.norm(Gamma[tsId].flatten())

plt.plot(y)
plt.show()
sys.exit()

# weights = ecoMfdNN.getNNWeights(ecoMfdNN.params)
# ecoMfdNN.nnModel.set_weights(weights=weights)
# inpVars = tf.Variable(ecoMfdNN.cumSol, dtype=tf.float32)
# GammaVec = ecoMfdNN.nnModel(inpVars).numpy()

#for item in weights:
#    print(np.linalg.norm(item))
#print(np.linalg.norm(GammaVec))
#val = ecoMfdNN.getObjFunc(ecoMfdNN.params)
#Gamma = ecoMfdNN.getGamma(ecoMfdNN.params)
#print("Gamma norm =", np.linalg.norm(Gamma.reshape(Gamma.size)))
#for tsId in range(ecoMfdNN.nTimes):
#    print("Gamma[%d] norm = %0.8f" % (tsId, np.linalg.norm(Gamma[tsId].reshape(Gamma[tsId].size))))
sol = ecoMfdNN.getSol(ecoMfdNN.params)
#grad = ecoMfdNN.getGrad(ecoMfdNN.params)
for i in range(ecoMfdNN.nParams):
    params1 = ecoMfdNN.params.copy()
    params1[i] += eps
    sol1 = ecoMfdNN.getSol(params1)
    tmp_sol = np.linalg.norm((sol-sol1).reshape(sol.size))    
    print("sol diff at %d = %0.8f" % (i+1, tmp_sol))
    

    # weights1 = ecoMfdNN.getNNWeights(params1)
    # ecoMfdNN.nnModel.set_weights(weights=weights1)
    # inpVars = tf.Variable(ecoMfdNN.cumSol, dtype=tf.float32)
    # GammaVec1 = ecoMfdNN.nnModel(inpVars).numpy()
#    for item in weights1:
#        print(np.linalg.norm(item))
#    print(np.linalg.norm(GammaVec1))
#    sys.exit()
#    print("norm Grad_gamma_vec[%d] = %0.8f" % (i+1, np.linalg.norm(GammaVec1-GammaVec)/eps))
    
#    val1 = ecoMfdNN.getObjFunc(params1)
#    Gamma1 = ecoMfdNN.getGamma(params1)    
#    sol1 = ecoMfdNN.getSol(params1)
#    print("Gamma1 norm =", np.linalg.norm(Gamma1.reshape(Gamma1.size)))    
#    sys.exit()
#    tmp_sol = np.linalg.norm((sol-sol1).reshape(sol.size))
#    tmp_gamma = np.linalg.norm((Gamma-Gamma1).reshape(Gamma.size))    
    
#    print(
#        "Grad[%d] = %0.8f; Gamma diff = %0.8f; sol diff = %0.8f" % (i+1, (val1-val)/eps, tmp_gamma, tmp_sol)
#    )
    
    
#sFlag = ecoMfdNN.setParams()
#print(sFlag)
#t0 = time.time()
#print(ecoMfdNN.getObjFunc(ecoMfdNN.params))
#print("Obj func:", (time.time() - t0))

#t0 = time.time()
#print(ecoMfdNN.getGamma(ecoMfdNN.params).flatten().mean())
#print(ecoMfdNN.getGrad(ecoMfdNN.params).flatten().mean())
#print("Grad:", (time.time() - t0))

#sFlag = ecoMfdNN.setParams()

#print(ecoMfdNN.params)
#print(ecoMfdNN.getGamma(ecoMfdNN.params))
#print(ecoMfdNN.cumSol[0][1], ecoMfdNN.cumSol[-1][1])
#assert sFlag

#ecoMfdNN.pltResults(rType="trn")

