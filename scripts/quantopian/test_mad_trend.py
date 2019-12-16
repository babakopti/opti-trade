import pandas as pd
import numpy as np
import quantopian.optimize as opt
import quantopian.algorithm as algo
from quantopian.algorithm import order_optimal_portfolio

wtHash_mad = {}

def initialize( context ):
    
    context.wtHash = wtHash_mad
    
    context.secList = [ symbol('QQQ'), symbol('SPY'), symbol('DIA'), symbol('MDY'), symbol('IWM'), symbol('OIH'), symbol('SMH'), symbol('XLE'), symbol('XLF'), symbol('XLU'), symbol('EWJ') ]
    
    context.weights = {}
    
    schedule_function( rebalance, 
                       date_rule = date_rules.every_day(), 
                       time_rule = time_rules.market_open() )

def rebalance( context, data ):
    
    curDate    = str( get_datetime().date() )
    
    #print( 'CurDate:', curDate )
    #print( context.wtHash.keys() )

    if curDate in list(context.wtHash.keys()):
    
        context.weights = {}
        
        for item in context.secList:
            symb = item.symbol
            if symb not in context.wtHash[curDate].keys():
                continue
            context.weights[item] = context.wtHash[curDate][symb] 
        
        print( 'Weights updated for date', curDate ) 
            
    order_optimal_portfolio( opt.TargetWeights( context.weights ), 
                             constraints = [] )
