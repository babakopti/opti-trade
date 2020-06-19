# ***********************************************************************
# Import libraries
# ***********************************************************************

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import timedelta

# ***********************************************************************
# Some Parameters
# ***********************************************************************

minProb  = 0.45
maxPrice = 2000.0
tradeFee = 0.75

# ***********************************************************************
# Put all pieces together
# ***********************************************************************

if False:
    df = pd.DataFrame()

    for file_name in os.listdir( 'models' ):
    
        if file_name.split('_')[0] != 'options':
            continue
    
        file_path = os.path.join('models', file_name)

        tmp_df = pd.read_pickle( file_path )
        
        df = pd.concat( [df, tmp_df ] )

    df.to_pickle( 'options_test_all.pkl' )
else:
    df = pd.read_pickle( 'options_test_all.pkl' )

df[ 'Year' ]     = df.DataDate.apply( lambda x : x.year )
df[ 'Bin_Prob' ] = df.Probability.apply( lambda x : 0.05 * int( x / 0.05 ) )
df[ 'horizon' ]  = df.horizon.apply( lambda x : int( x.days ) )

# ***********************************************************************
# Analyze call options
# ***********************************************************************
    
call_df = df[ df.Type == 'call' ]

call_df[ 'Return' ] = ( call_df[ 'actExprPrice' ] - \
                        call_df[ 'Strike' ] - \
                        call_df[ 'Last' ]  - tradeFee ) / call_df[ 'Last' ]

call_df[ 'Return' ] = call_df[ 'Return' ].\
    apply( lambda x : max( x, -1 ) )

call_df['Success'] = call_df[ 'Return' ].\
    apply( lambda x: True if x > 0 else False )

ch_call_df = call_df[ call_df.Probability > minProb ]
lim_ch_call_df = ch_call_df[ ch_call_df.Last < maxPrice / 100.0 ]

print( 'Call success / probability summary:',
       call_df.groupby( 'Success' )[ 'Probability' ].mean() )

print( 'Overall call success rate:',
       call_df[ call_df.Success ].shape[0] / call_df.shape[0] )

print( 'Chosen call success rate:',
       ch_call_df[ ch_call_df.Success ].shape[0] / ch_call_df.shape[0] )

print( 'Overall call average return:',
       call_df.Return.mean() )

print( 'Chosen call average return:',
       ch_call_df.Return.mean() )

print( 'Overall call median return:',
       call_df.Return.median() )

print( 'Chosen call median return:',
       ch_call_df.Return.median() )

print( 'Overall call average return summary:',
       call_df.groupby( 'Year' )[ 'Return' ].mean() )

print( 'Chosen call average return summary:',
       ch_call_df.groupby( 'Year' )[ 'Return' ].mean() )

print( 'Overall call median return summary:',
       call_df.groupby( 'Year' )[ 'Return' ].median() )

print( 'Chosen call median return summary:',
       ch_call_df.groupby( 'Year' )[ 'Return' ].median() )

print( 'Chosen call count with price limit:',
    ch_call_df[ ch_call_df.Last < maxPrice / 100.0 ].\
       groupby( 'Year' )[ 'OptionSymbol' ].count() )

print( 'Chosen call median return summary with price limit:',
    ch_call_df[ ch_call_df.Last < maxPrice / 100.0 ].\
       groupby( 'Year' )[ 'Return' ].median() )

if False:
    plt.scatter( ch_call_df.Probability, ch_call_df.Return )
    plt.title( 'Return vs. probability for call options!' )
    plt.xlabel( 'Probability' )
    plt.ylabel( 'Return' )
    plt.show()

    plt.scatter( ch_call_df.Probability, 100 * ch_call_df.Last )
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

plt.plot( plt_df.Bin_Prob, plt_df.horizon, 'b-o',
          lim_plt_df.Bin_Prob, lim_plt_df.horizon, 'r-o' )
plt.title( 'Horizon vs. binned probability for call options!' )
plt.legend( [ 'All', 'Limited price' ] )
plt.xlabel( 'Probability' )
plt.ylabel( 'Horizon (days)' )
plt.show()

plt.plot( plt_df.Bin_Prob, 100 * plt_df.Last, '-o' )
plt.title( 'Contract price vs. binned probability for call options!' )
plt.xlabel( 'Probability' )
plt.ylabel( 'Price' )
plt.show()

# ***********************************************************************
# Analyze put options
# ***********************************************************************

put_df = df[ df.Type == 'put' ]

put_df[ 'Return' ] = ( -put_df[ 'actExprPrice' ] + \
                        put_df[ 'Strike' ] - \
                        put_df[ 'Last' ]  - tradeFee ) / put_df[ 'Last' ]

put_df[ 'Return' ] = put_df[ 'Return' ].\
    apply( lambda x : max( x, -1 ) )

put_df['Success'] = put_df[ 'Return' ].\
    apply( lambda x: True if x > 0 else False )

ch_put_df = put_df[ put_df.Probability > minProb ]
lim_ch_put_df = ch_put_df[ ch_put_df.Last < maxPrice / 100.0 ]

print( 'Put success / probability summary:',
       put_df.groupby( 'Success' )[ 'Probability' ].mean() )

print( 'Overall put success rate:',
       put_df[ put_df.Success ].shape[0] / put_df.shape[0] )

print( 'Chosen put success rate:',
       ch_put_df[ ch_put_df.Success ].shape[0] / ch_put_df.shape[0] )

print( 'Overall put average return:',
       put_df.Return.mean() )

print( 'Chosen put average return:',
       ch_put_df.Return.mean() )

print( 'Overall put median return:',
       put_df.Return.median() )

print( 'Chosen put median return:',
       ch_put_df.Return.median() )

print( 'Overall put average return summary:',
       put_df.groupby( 'Year' )[ 'Return' ].mean() )

print( 'Chosen put average return summary:',
       ch_put_df.groupby( 'Year' )[ 'Return' ].mean() )

print( 'Overall put median return summary:',
       put_df.groupby( 'Year' )[ 'Return' ].median() )

print( 'Chosen put median return summary:',
       ch_put_df.groupby( 'Year' )[ 'Return' ].median() )

print( 'Chosen put count with price limit:',
    ch_put_df[ ch_put_df.Last < maxPrice / 100.0 ].\
       groupby( 'Year' )[ 'OptionSymbol' ].count() )

print( 'Chosen put median return summary with price limit:',
       ch_put_df[ ch_put_df.Last < maxPrice / 100.0 ].\
       groupby( 'Year' )[ 'Return' ].median() )

if False:
    plt.scatter( ch_put_df.Probability, ch_put_df.Return )
    plt.title( 'Return vs. probability for put options!' )
    plt.xlabel( 'Probability' )
    plt.ylabel( 'Return' )
    plt.show()

    plt.scatter( ch_put_df.Probability, 100 * ch_put_df.Last )
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

plt.plot( plt_df.Bin_Prob, plt_df.horizon, 'b-o',
          lim_plt_df.Bin_Prob, lim_plt_df.horizon, 'r-o' )
plt.title( 'Horizon vs. binned probability for put options!' )
plt.legend( [ 'All', 'Limited price' ] )
plt.xlabel( 'Probability' )
plt.ylabel( 'Horizon (days)' )
plt.show()

plt.plot( plt_df.Bin_Prob, 100 * plt_df.Last, '-o' )
plt.title( 'Contract price vs. binned probability for put options!' )
plt.xlabel( 'Probability' )
plt.ylabel( 'Price' )
plt.show()

