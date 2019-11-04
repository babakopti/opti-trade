
import talib
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import quantopian.optimize as opt
import quantopian.algorithm as algo
from quantopian.algorithm import order_optimal_portfolio

OPT_TOL   = 1.0e-6

stgWtHash = { 'MACD' : 0.6, 
              'MSDP' : 0.1, 
              'MSDV' : 0.1, 
              'RLS'  : 0.2   } 

def getMACDTrend( asset, data ):
    
    def MACD( prices, 
              fastperiod   = 12, 
              slowperiod   = 26, 
              signalperiod = 9    ):

        macd, signal, hist = talib.MACD(prices,
                                        fastperiod=fastperiod,
                                        slowperiod=slowperiod,
                                        signalperiod=signalperiod)
        return macd[-1] - signal[-1]
    
    prices = data.history( asset, 'price', 40, '1d' )
    prices = np.array( prices )
    macd   = MACD( prices ) 
 
    if macd > 0:
       trend = 1.0
    elif macd < 0:
       trend = -1.0
    else:
       trend = 0.0
        
    return trend
        
def getMSDPTrend( asset, data ):
    
    price_history = data.history( asset, 'price', 20, '1d' )
    current_price = data.current( asset, 'price' )
    mean          = price_history.mean()
    stddev        = price_history.std()
 
    if current_price < mean - 1.75 * stddev:
        trend = 1.0
    elif current_price > mean + 1.75 * stddev:
        trend = -1.0
    else:
        trend = 0.0
        
    return trend

def getMSDVTrend( asset, data ):
    
    volume_history = data.history( asset, 'volume', 20, '1d' )
    current_volume = data.current( asset, 'volume' )
    mean          = volume_history.mean()
    stddev        = volume_history.std()
 
    if current_volume < mean - 1.75 * stddev:
        trend = 1.0
    elif current_volume > mean + 1.75 * stddev:
        trend = -1.0
    else:
        trend = 0.0
        
    return trend

def getRLSTrend( asset, data ):
    
    spy    = symbol('SPY')
    spyVec = data.history( spy, 'price', 20, '1d' )
    vec    = data.history( asset, 'price', 20, '1d' )
    spyVec = np.array( spyVec )
    vec    = np.array( vec )
    
    assert len( spyVec ) == len( vec ), \
    'Inconsistent size! %d vs. %d' % ( len(spyVec), len(vec) )
    
    rlsVec = np.zeros( shape = ( len(vec) ), dtype = 'd' )
    
    for i in range( len(vec) ):
        rlsVec[i] = vec[i] / spyVec[i]
    
    current_rls = rlsVec[-1]
    mean        = np.mean( rlsVec )
    stddev      = np.std( rlsVec )
 
    if current_rls < mean - 1.75 * stddev:
        trend = 1.0
    elif current_rls > mean + 1.75 * stddev:
        trend = -1.0
    else:
        trend = 0.0
        
    return trend

def getConsGuesses( context, data ):
    
    cons = []    
    guesses = []
    sumFunc = lambda wts : ( sum( abs( wts ) ) - 1.0 )
    cons.append( { 'type' : 'eq', 'fun' : sumFunc } )

    for i in range( len(context.assets) ):
        
        asset = context.assets[i]
        
        macdTrend = getMACDTrend( asset, data )
        msdpTrend = getMSDPTrend( asset, data )
        msdvTrend = getMSDVTrend( asset, data )
        rlsTrend  = getRLSTrend( asset, data )
        
        probLong  = 0
        probShort = 0
        
        if macdTrend > 0:
            probLong  += stgWtHash[ 'MACD' ] 
        elif macdTrend < 0:
            probShort += stgWtHash[ 'MACD' ] 
            
        if msdpTrend > 0:
            probLong  += stgWtHash[ 'MSDP' ] 
        elif msdpTrend < 0:
            probShort += stgWtHash[ 'MSDP' ] 
            
        if msdvTrend > 0:
            probLong  += stgWtHash[ 'MSDV' ] 
        elif msdvTrend < 0:
            probShort += stgWtHash[ 'MSDV' ] 
            
        if rlsTrend > 0:
            probLong  += stgWtHash[ 'RLS' ] 
        elif rlsTrend < 0:
            probShort += stgWtHash[ 'RLS' ] 
            
        if probLong > 0.8:
            trendFunc = lambda wts : wts[i]
            cons.append( { 'type' : 'ineq', 'fun' : trendFunc } )
            guesses.append( 1.0 )
        elif probShort > 0.8:
            trendFunc = lambda wts : -wts[i]
            cons.append( { 'type' : 'ineq', 'fun' : trendFunc } )
            guesses.append( -1.0 )
        else:
            guesses.append( 0.0 )
            
    return cons, guesses

def checkCons( cons, wts ):

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

def minimum_MAD_portfolio( context, data ):
 
    def _mad(x, returns):
        return (returns - returns.mean()).dot(x).abs().mean()
    
    cons, guesses = getConsGuesses( context, data )
    
    returns = np.log( data.history( context.assets, 
                                    'price', 
                                    200, 
                                    '1d') ).pct_change().dropna()

    results   = minimize( fun         = _mad,
                          x0          = guesses,
                          args        = returns, 
                          constraints = cons,
                          method      = 'SLSQP',
                          tol         = OPT_TOL,
                          options     = { 'maxiter' : 1000 } )
    
    checkCons( cons, results.x )
    
    return pd.Series(index=returns.columns, data=results.x)

def initialize(context):

    set_commission(commission.PerTrade(cost=0))
    
    set_slippage(slippage.FixedSlippage(spread=0))
    
    context.assets = [ symbol('QQQ'), 
                       symbol('SPY'), 
                       symbol('DIA'), 
                       symbol('MDY'), 
                       symbol('IWM'), 
                       symbol('OIH'), 
                       symbol('SMH'), 
                       symbol('XLE'), 
                       symbol('XLF'), 
                       symbol('XLU'), 
                       symbol('EWJ') ]
    
    context.assets = [ symbol('FAS'), 
                       symbol('TNA'), 
                       symbol('TQQQ'), 
                       symbol('EDC'), 
                       symbol('LABU') ]
    
    schedule_function(rebalance, date_rules.every_day())

def rebalance(context, data):
    
    weights = minimum_MAD_portfolio( context, data )
 
    for security in context.assets:
        print(weights[security])
        order_target_percent(security, weights[security])
