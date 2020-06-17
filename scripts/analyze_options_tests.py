# ***********************************************************************
# Import libraries
# ***********************************************************************

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ***********************************************************************
# Some Parameters
# ***********************************************************************

minProb = 0.55

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

# ***********************************************************************
# Analyze call options
# ***********************************************************************
    
call_df = df[ df.Type == 'call' ]

call_df[ 'Return' ] = ( call_df[ 'actExprPrice' ] - \
                        call_df[ 'Strike' ] - \
                        call_df[ 'Last' ]  ) / call_df[ 'Last' ]

call_df[ 'Return' ] = call_df[ 'Return' ].\
    apply( lambda x : max( x, -1 ) )

call_df['Success'] = call_df[ 'Return' ].\
    apply( lambda x: True if x > 0 else False )

ch_call_df = call_df[ call_df.Probability > minProb ]

print( 'Call success / probability summary:',
       call_df.groupby( 'Success' )[ 'Probability' ].mean() )

print( 'Overall call success rate:',
       call_df[ call_df.Success ].shape[0] / call_df.shape[0] )

print( 'Chosen call success rate:',
       ch_call_df[ ch_call_df.Success ].shape[0] / ch_call_df.shape[0] )

print( 'Overall call average return:',
       call_df['Return'].mean() )

print( 'Chosen call average return:',
       ch_call_df['Return'].mean() )

print( 'Overall call median return:',
       call_df['Return'].median() )

print( 'Chosen call median return:',
       ch_call_df['Return'].median() )

plt.scatter( ch_call_df.Probability, ch_call_df.Return )
plt.title( 'Return vs. probability for call options!' )
plt.xlabel( 'Probability' )
plt.ylabel( 'Return' )
plt.show()

# ***********************************************************************
# Analyze put options
# ***********************************************************************

put_df = df[ df.Type == 'put' ]

put_df[ 'Return' ] = ( -put_df[ 'actExprPrice' ] + \
                        put_df[ 'Strike' ] - \
                        put_df[ 'Last' ]  ) / put_df[ 'Last' ]

put_df[ 'Return' ] = put_df[ 'Return' ].\
    apply( lambda x : max( x, -1 ) )

put_df['Success'] = put_df[ 'Return' ].\
    apply( lambda x: True if x > 0 else False )

ch_put_df = put_df[ put_df.Probability > minProb ]

print( 'Put success / probability summary:',
       put_df.groupby( 'Success' )[ 'Probability' ].mean() )

print( 'Overall put success rate:',
       put_df[ put_df.Success ].shape[0] / put_df.shape[0] )

print( 'Chosen put success rate:',
       ch_put_df[ ch_put_df.Success ].shape[0] / ch_put_df.shape[0] )

print( 'Overall put average return:',
       put_df['Return'].mean() )

print( 'Chosen put average return:',
       ch_put_df['Return'].mean() )

print( 'Overall put median return:',
       put_df['Return'].median() )

print( 'Chosen put median return:',
       ch_put_df['Return'].median() )

plt.scatter( ch_put_df.Probability, ch_put_df.Return )
plt.title( 'Return vs. probability for put options!' )
plt.xlabel( 'Probability' )
plt.ylabel( 'Return' )
plt.show()



