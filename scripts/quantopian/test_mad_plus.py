# Minimum date inclusive

MIN_DATE = '2019-10-31'

# Maximum date inclusive
MAX_DATE = '2019-10-31'

# Weghts of different strategies
stgWtHash = { 'MACD' : 0.6, 
              'MSDP' : 0.1, 
              'MSDV' : 0.1, 
              'RLS'  : 0.2   } 

# List of securities to trade
security_universe = [ symbol('FAS'), 
                      symbol('TNA'), 
                      symbol('TQQQ'), 
                      symbol('EDC'), 
                      symbol('LABU') ]

# Interval between daily trades in minutes
INTERVAL = 45

# Total minutes in one trading day
TOTAL_MINUTES = 390 

# Start so many hours after market opens; applicabla only when interval = 0
START_HOUR = 6 

# Thershold probability for alllowing a trade
THRESHOLD_PROB = 0.6
# optimizer tolerance
OPT_TOL = 1.0e-6 

# ***********************************************************
# Import libs
# ***********************************************************

import talib
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import quantopian.optimize as opt
import quantopian.algorithm as algo
from quantopian.algorithm import order_optimal_portfolio

# ***********************************************************
# Some utlity functions
# ***********************************************************

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
    
    context.assets = []
    trends         = []
    guesses        = []
    offset         = 0.0
    
    for i in range( len(context.pool) ):
        
        asset = context.pool[i]
        
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
            
        if probLong > THRESHOLD_PROB:
            trends.append( 1.0 )
            guesses.append( 1.0 )
            context.assets.append( asset )
        elif probShort > THRESHOLD_PROB:
            trends.append( -1.0 )
            guesses.append( -1.0 )
            context.assets.append( asset )       
        else:
            offset += abs(context.weights[asset])
            print( 'No transaction recommended on', asset.symbol)
    
    assert len(trends) == len(guesses), 'Inconsistent sizes!'
    assert len(trends) == len(context.assets), 'Inconsistent sizes!'
    
    cons = []    
    sumFunc = lambda wts : ( sum( abs( wts ) ) + offset - 1.0 )
    cons.append( { 'type' : 'eq', 'fun' : sumFunc } )
    for i in range( len(context.assets) ):
        
        asset = context.assets[i]
        trend = trends[i]
            
        if trend > 0:
            trendFunc = lambda wts : wts[i] 
            cons.append( { 'type' : 'ineq', 'fun' : trendFunc } )
        elif trend < 0:
            trendFunc = lambda wts : -wts[i]
            cons.append( { 'type' : 'ineq', 'fun' : trendFunc } )
        else:
            assert False, 'Internal error: zero trend!'
            
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
    
    if len(context.assets) == 0:
        return context.weights
    
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

# ***********************************************************
# Algorithm
# ***********************************************************

def initialize(context):

    set_commission(commission.PerTrade(cost=0))
    
    set_slippage(slippage.FixedSlippage(spread=0))
        
    context.pool =  security_universe
    
    context.assets = []
    context.weights = {}
    for asset in context.pool:
        context.weights[asset] = 0
    
    if INTERVAL == 0:
        schedule_function( rebalance, 
                           date_rules.every_day(), 
                           time_rules.market_open(hours=START_HOUR,
                                                  minutes=1) )
    elif INTERVAL > 0:
        for minute in range(1, TOTAL_MINUTES, INTERVAL):  
            schedule_function( rebalance, 
                               date_rules.every_day(), 
                               time_rules.market_open(minutes=minute))
    else:
        assert False, 'INTERAVL should be positive!'
    
def handle_data(context, data):
    return
    #record(cash = context.portfolio.cash)
    record(value = context.portfolio.portfolio_value)
    
def rebalance(context, data):
    
    if str(get_datetime().date()) < MIN_DATE:
        return
    
    if str(get_datetime().date()) > MAX_DATE:
        for asset in context.pool:
            order_target(asset, 0)
        return
    
    weights = minimum_MAD_portfolio( context, data )
 
    for asset in context.assets:
        context.weights[asset] = weights[asset]
    
    for asset in context.weights.keys():
        print(asset.symbol,
              context.weights[asset],
              context.portfolio.positions[asset].amount)
        order_target_percent(asset, context.weights[asset])
