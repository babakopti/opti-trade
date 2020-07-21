import pandas as pd

OPTION_CHAIN_FILE = '/var/option_data/option_chain_2020-07.pkl'
ACT_FILE = '/var/option_data/dfFile_2020-07-21 10:45:41.pkl'

optDf = pd.read_pickle( OPTION_CHAIN_FILE )
actDf = pd.read_pickle( ACT_FILE )

actDf = actDf.melt( id_vars    = [ 'Date' ],
                    value_vars = list( set( actDf.columns ) - \
                                       { 'Date' } ) )

actDf[ 'Date' ] = actDf.Date.apply( lambda x : pd.to_datetime(x) )

actDf = actDf.rename( columns = { 'Date'     : 'expiration',
                                  'variable' : 'assetSymbol',
                                  'value'    : 'actExprPrice' }   )

optDf[ 'expiration' ] = optDf.expiration.apply( lambda x : pd.to_datetime(x) )

optDf = optDf.merge( actDf,
                     how = 'left',
                     on  = [ 'assetSymbol', 'expiration' ] )

