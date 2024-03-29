# ***********************************************************************
# Import libraries
# ***********************************************************************

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import timedelta
from collections import defaultdict

# ***********************************************************************
# Some Parameters
# ***********************************************************************

minProb  = 0.5
maxPrice = 20000.0
maxHoldA = 20000000.0
tradeFee = 0.75

minHorizon = 1
maxHorizon = 10

OPTION_CHAIN_FILE = 'options_chain_data/option_chain_Aug_Nov_2020.pkl'
ACT_FILE = 'data/dfFile_2020-11-10 15:00:06.pkl'

# ***********************************************************************
# Read options and actuals files and merge
# ***********************************************************************

optDf = pd.read_pickle( OPTION_CHAIN_FILE )
    
if True:
    actDf = pd.read_pickle( ACT_FILE )

    actDf[ 'Date' ] = actDf.Date.astype( 'datetime64[ns]' )
    actDf[ 'Date' ] = actDf.Date.apply( lambda x: x.strftime( '%Y-%m-%d' ) )
    actDf[ 'Date' ] = actDf.Date.astype( 'datetime64[ns]' )
    
    actDf = actDf.groupby( 'Date', as_index = False ).mean()
    
    actDf = actDf.melt( id_vars    = [ 'Date' ],
                        value_vars = list( set( actDf.columns ) - \
                                           { 'Date' } ) )

    actDf = actDf.rename( columns = { 'Date'     : 'expiration',
                                      'variable' : 'assetSymbol',
                                      'value'    : 'actExprPrice' }   )

    optDf[ 'expiration' ] = optDf.expiration.astype( 'datetime64[ns]' )
    optDf[ 'DataDate' ]  = optDf.DataDate.astype( 'datetime64[ns]' )

    df = optDf.merge( actDf,
                      how = 'left',
                      on  = [ 'assetSymbol', 'expiration' ] )
    df = df.dropna()

    df[ 'horizon' ]  = df.expiration - df.DataDate    
    df[ 'Year' ]     = df.DataDate.apply( lambda x : x.year )
    df[ 'Bin_Prob' ] = df.Prob.apply( lambda x : 0.05 * int( x / 0.05 ) )
    df[ 'horizon' ]  = df.horizon.apply( lambda x : int( x.days ) )
    df[ 'strike' ]   = df.strike.apply( lambda x : float( x ) )
    
# ***********************************************************************
# Study call options
# ***********************************************************************

call_df = df[ df[ 'type' ] == 'call' ]

call_df[ 'Return' ] = ( call_df[ 'actExprPrice' ] - \
                        call_df[ 'strike' ] - \
                        call_df[ 'unitPrice' ]  - tradeFee / 100 ) / call_df[ 'unitPrice' ]

call_df[ 'Return' ] = call_df[ 'Return' ].\
    apply( lambda x : max( x, -1 ) )

call_df[ 'Success' ] = call_df[ 'Return' ].\
    apply( lambda x: 1 if x > 0 else 0 )

ch_call_df = call_df[ call_df.Prob > minProb ]
lim_ch_call_df = ch_call_df[ ch_call_df.unitPrice < maxPrice / 100.0 ]

print( 'Call success / probability summary:',
       call_df.groupby( 'Success' )[ 'Prob' ].mean() )

print( 'Overall call success rate:',
       call_df[ call_df.Success == 1 ].shape[0] / call_df.shape[0] )

if ch_call_df.shape[0] > 0:
    print( 'Chosen call success rate:',
           ch_call_df[ ch_call_df.Success == 1 ].shape[0] / ch_call_df.shape[0] )

print( 'Overall call average return:',
       call_df.Return.mean() )

print( 'Chosen call average return:',
       ch_call_df.Return.mean() )

print( 'Overall call median return:',
       call_df.Return.median() )

print( 'Chosen call median return:',
       ch_call_df.Return.median() )

tmp_list_call = []
tmp_df = call_df[ (call_df.Prob > minProb) & \
                  (100 * call_df.unitPrice < maxPrice) & \
                  (call_df.horizon >= minHorizon) & \
                  (call_df.horizon <= maxHorizon) ]

for itr in range( 1000 ):
    
    holdHash = defaultdict( float )
    returns = []    
    nSelected = 0
    
    for i in range( tmp_df.shape[0] ):
        
        item = tmp_df.sample( n = 1,
                              replace = True,
                              weights = 'Prob',
                              axis = 0 )
        
        assetSymbol = list(item.assetSymbol)[0]
        price = 100 * list(item.unitPrice)[0]

        if price + holdHash[ assetSymbol ] <= maxHoldA:
            nSelected += 1
            holdHash[ assetSymbol ] += price
            returns.append( list(item.Return)[0] )
        else:
            continue

        if nSelected >= 10:
            break
    
    tmp_list_call.append( np.mean( returns ) )

print( 'Chosen call count/avg_horizon/min/max/mean/std from monte-carlo: '
       '%d, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f' % (
           tmp_df.shape[0],
           tmp_df.horizon.mean(),
           np.min( tmp_list_call ),
           np.max( tmp_list_call ),
           np.mean( tmp_list_call ),
           np.std( tmp_list_call ) ) )

print('Summary of selected call options by asset symbol:')
print(ch_call_df.groupby(['Year', 'assetSymbol'])['Return', 'Prob'].mean())

if False:
    sns.distplot(call_df.Return);
    sns.distplot(ch_call_df.Return);
    plt.legend(['All', 'Chosen'])
    plt.xlabel( 'Return' )
    plt.ylabel( 'Distribution (call options)' )
    plt.show()

if False:
    plt.scatter( ch_call_df.Prob, ch_call_df.Return )
    plt.title( 'Return vs. probability for call options!' )
    plt.xlabel( 'Probability' )
    plt.ylabel( 'Return' )
    plt.show()

    plt.scatter( ch_call_df.Prob, 100 * ch_call_df.unitPrice )
    plt.title( 'Contract price vs. probability for call options!' )
    plt.xlabel( 'Probability' )
    plt.ylabel( 'Price' )
    plt.show()

    plt_df = ch_call_df.groupby( 'Bin_Prob', as_index = False ).mean()
    lim_plt_df = lim_ch_call_df.groupby( 'Bin_Prob', as_index = False ).mean()

    plt.plot( plt_df.Bin_Prob, plt_df.Return, 'b-o',
              lim_plt_df.Bin_Prob, lim_plt_df.Return, 'r-o' )
    plt.title( 'Return vs. binned probability for call options!' )
    plt.legend( [ 'All', 'Limited price' ] )
    plt.xlabel( 'Probability' )
    plt.ylabel( 'Return' )
    plt.show()

    plt.plot( plt_df.Bin_Prob, plt_df.Success, 'b-o',
              lim_plt_df.Bin_Prob, lim_plt_df.Success, 'r-o' )
    plt.title( 'Success rate vs. binned probability for call options!' )
    plt.legend( [ 'All', 'Limited price' ] )
    plt.xlabel( 'Probability' )
    plt.ylabel( 'Success rate' )
    plt.show()

    plt.plot( plt_df.Bin_Prob, plt_df.horizon, 'b-o',
              lim_plt_df.Bin_Prob, lim_plt_df.horizon, 'r-o' )
    plt.title( 'Horizon vs. binned probability for call options!' )
    plt.legend( [ 'All', 'Limited price' ] )
    plt.xlabel( 'Probability' )
    plt.ylabel( 'Horizon (days)' )
    plt.show()

    plt.plot( plt_df.Bin_Prob, 100 * plt_df.unitPrice, '-o' )
    plt.title( 'Contract price vs. binned probability for call options!' )
    plt.xlabel( 'Probability' )
    plt.ylabel( 'Price' )
    plt.show()

# ***********************************************************************
# Analyze put options
# ***********************************************************************

put_df = df[ df[ 'type' ] == 'put' ]

put_df[ 'Return' ] = ( -put_df[ 'actExprPrice' ] + \
                        put_df[ 'strike' ] - \
                        put_df[ 'unitPrice' ]  - tradeFee / 100 ) / put_df[ 'unitPrice' ]

put_df[ 'Return' ] = put_df[ 'Return' ].\
    apply( lambda x : max( x, -1 ) )

put_df['Success'] = put_df[ 'Return' ].\
    apply( lambda x: 1 if x > 0 else 0 )

ch_put_df = put_df[ put_df.Prob > minProb ]
lim_ch_put_df = ch_put_df[ ch_put_df.unitPrice < maxPrice / 100.0 ]

print( 'Put success / probability summary:',
       put_df.groupby( 'Success' )[ 'Prob' ].mean() )

print( 'Overall put success rate:',
       put_df[ put_df.Success == 1 ].shape[0] / put_df.shape[0] )

if ch_put_df.shape[0] > 0:
    print( 'Chosen put success rate:',
           ch_put_df[ ch_put_df.Success == 1 ].shape[0] /  ch_put_df.shape[0])

print( 'Overall put average return:',
       put_df.Return.mean() )

print( 'Chosen put average return:',
       ch_put_df.Return.mean() )

print( 'Overall put median return:',
       put_df.Return.median() )

print( 'Chosen put median return:',
       ch_put_df.Return.median() )

tmp_list_put = []
tmp_df = put_df[ (put_df.Prob > minProb) & \
                 (100 * put_df.unitPrice < maxPrice) & \
                  (put_df.horizon >= minHorizon) & \
                  (put_df.horizon <= maxHorizon) ]
                 

for itr in range( 1000 ):
    
    holdHash = defaultdict( float )
    returns = []    
    nSelected = 0
    
    for i in range( tmp_df.shape[0] ):
        
        item = tmp_df.sample( n = 1,
                              replace = True,
                              weights = 'Prob',
                              axis = 0 )
        
        assetSymbol = list(item.assetSymbol)[0]
        price = 100 * list(item.unitPrice)[0]

        if price + holdHash[ assetSymbol ] <= maxHoldA:
            nSelected += 1
            holdHash[ assetSymbol ] += price
            returns.append( list(item.Return)[0] )
        else:
            continue

        if nSelected >= 10:
            break
    
    tmp_list_put.append( np.mean( returns ) )

print( 'Chosen put count/avg. horizon/min/max/mean/std from monte-carlo: '
       '%d, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f' % (
           tmp_df.shape[0],
           tmp_df.horizon.mean(),
           np.min( tmp_list_put ),
           np.max( tmp_list_put ),
           np.mean( tmp_list_put ),
           np.std( tmp_list_put ) ) )

print('Summary of selected put options by asset symbol:')
print(ch_put_df.groupby(['Year', 'assetSymbol'])['Return', 'Prob'].mean())

if False:
    sns.distplot(put_df.Return)
    sns.distplot(ch_put_df.Return)
    plt.legend(['All', 'Chosen'])
    plt.xlabel( 'Return' )
    plt.ylabel( 'Distribution (put options)' )
    plt.show()

if False:
    plt.scatter( ch_put_df.Prob, ch_put_df.Return )
    plt.title( 'Return vs. probability for put options!' )
    plt.xlabel( 'Probability' )
    plt.ylabel( 'Return' )
    plt.show()

    plt.scatter( ch_put_df.Prob, 100 * ch_put_df.unitPrice )
    plt.title( 'Contract price vs. probability for put options!' )
    plt.xlabel( 'Probability' )
    plt.ylabel( 'Price' )
    plt.show()

    plt_df = ch_put_df.groupby( 'Bin_Prob', as_index = False ).mean()
    lim_plt_df = lim_ch_put_df.groupby( 'Bin_Prob', as_index = False ).mean()            

    plt.plot( plt_df.Bin_Prob, plt_df.Return, 'b-o',
              lim_plt_df.Bin_Prob, lim_plt_df.Return, 'r-o' )
    plt.title( 'Return vs. binned probability for put options!' )
    plt.legend( [ 'All', 'Limited price' ] )
    plt.xlabel( 'Probability' )
    plt.ylabel( 'Return' )
    plt.show()

    plt.plot( plt_df.Bin_Prob, plt_df.Success, 'b-o',
              lim_plt_df.Bin_Prob, lim_plt_df.Success, 'r-o' )
    plt.title( 'Success rate vs. binned probability for put options!' )
    plt.legend( [ 'All', 'Limited price' ] )
    plt.xlabel( 'Probability' )
    plt.ylabel( 'Success rate' )
    plt.show()

    plt.plot( plt_df.Bin_Prob, plt_df.horizon, 'b-o',
              lim_plt_df.Bin_Prob, lim_plt_df.horizon, 'r-o' )
    plt.title( 'Horizon vs. binned probability for put options!' )
    plt.legend( [ 'All', 'Limited price' ] )
    plt.xlabel( 'Probability' )
    plt.ylabel( 'Horizon (days)' )
    plt.show()

    plt.plot( plt_df.Bin_Prob, 100 * plt_df.unitPrice, '-o' )
    plt.title( 'Contract price vs. binned probability for put options!' )
    plt.xlabel( 'Probability' )
    plt.ylabel( 'Price' )
    plt.show()

