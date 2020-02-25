# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import requests
import tdameritrade
import tdameritrade.auth as auth

from tdameritrade import TDClient

sys.path.append( os.path.abspath( '../' ) )

from utl.utils import getLogger

# ***********************************************************************
# Class Tdam: A class to trade with TD Ameritrade
# ***********************************************************************

class Tdam:

    def __init__(   self,
                    authKey      = None,
                    refToken     = None,
                    callbackUrl  = 'http://localhost:8080',
                    logFileName  = None,                    
                    verbose      = 1          ):

        if authKey is None:
            self.authKey  = 'QKSFXGSPRYAV3FMQMKTA23WXSDGSJL2A'
        else:
            self.authKey  = authKey

        self.refToken    = refToken

        self.callbackUrl = callbackUrl
        self.logFileName = logFileName
        self.verbose     = verbose
        self.logger      = getLogger( logFileName, verbose, 'tdam' )
        
        self.token       = None
        self.client      = None
        self.accounts    = None
        self.accountId   = None

        self.setAuth()
        self.setAccounts()
        self.setAccountId()

    def setAuth( self ):

        if self.refToken is None:
            accessHash = auth.authentication( client_id    = self.authKey,
                                              redirect_uri = self.callbackUrl )
            self.refToken = accessHash[ 'refresh_token' ]
        else:
            accessHash = auth.refresh_token( refresh_token = self.refToken,
                                             client_id     = self.authKey     )
        
        self.token  = accessHash[ 'access_token' ]
        self.client = TDClient( self.token )
        
    def setAccounts( self ):

        self.accounts = self.client.accounts()

        assert len( self.accounts.keys() ) > 0, 'No accounts found!'

    def setAccountId( self, accountId = None ):

        if accountId is None:
            self.accountId = list( self.accounts.keys() )[0]
        else:
            self.accountId = str( accountId )
        
    def getQuote( self, symbol, pType = 'ask' ):

        if pType == 'ask':
            item = 'askPrice'
        elif ptype == 'bid':
            item = 'bidPrice'
        else:
            item = 'lastPrice'
            
        price = self.client.quote( symbol )[ symbol ][ item ]

        return price

    def getOptionsChain( self, symbol ):

        topHash   = self.client.options( symbol )
        callHash  = topHash[ 'callExpDateMap' ]
        putHash   = topHash[ 'callExpDateMap' ]
        options   = []
        
        for date in callHash:
            
            stkHash  = callHash[ date ]
            exprDate = date.split( ':' )[0]
            
            for strike in stkHash:
                
                obj = stkHash[ strike ][0]
                
                assert symbol == obj[ 'symbol' ].split( '_' )[0], \
                    'Inconsistent underlying symbol %s vs. %s!' % \
                    ( symbol, obj[ 'symbol' ].split( '_' )[0] ) 

                assert float( strike ) == float( obj[ 'strikePrice' ] ), \
                    'Inconsistent strikePrices %s vs. %s!' % \
                    ( str( strike ), str( obj[ 'strikePrice' ] ) )

                option = { 'optionSymbol' : obj[ 'symbol' ],
                           'assetSymbol'  : symbol,
                           'strike'       : strike,
                           'expiration'   : exprDate,
                           'contractCnt'  : obj[ 'multiplier' ],                     
                           'unitPrice'    : obj[ 'ask' ],
                           'type'         : 'call'      }
                
                options.append( option )

        for date in putHash:
            
            stkHash  = putHash[ date ]
            exprDate = date.split( ':' )[0]
            
            for strike in stkHash:
                
                obj = stkHash[ strike ][0]

                assert symbol == obj[ 'symbol' ].split( '_' )[0], \
                    'Inconsistent underlying symbol %s vs. %s!' % \
                    ( symbol, obj[ 'symbol' ].split( '_' )[0] ) 

                assert float( strike ) == float( obj[ 'strikePrice' ] ), \
                    'Inconsistent strikePrices %s vs. %s!' % \
                    ( str( strike ), str( obj[ 'strikePrice' ] ) )
                
                option = { 'optionSymbol' : obj[ 'symbol' ],
                           'assetSymbol'  : symbol,
                           'strike'       : strike,
                           'expiration'   : exprDate,
                           'contractCnt'  : obj[ 'multiplier' ],                     
                           'unitPrice'    : obj[ 'ask' ],
                           'type'         : 'put'      }
                
                options.append( option )
                
        return options

    def order( self,
               symbol,
               orderType = 'MARKET',
               duration  = 'DAY',
               price     = None,
               quantity  = 1,
               sType     = 'EQUITY',
               action    = 'BUY' ):

        self.logger.info( 'Ordering %d of %s...', quantity, symbol )

        assert orderType in [ 'MARKET', 'LIMIT' ], 'Incorrect orderType!'
        assert duration in  [ 'DAY',
                              'GOOD_TILL_CANCEL',
                              'FILL_OR_KILL' ], 'Incorrect duration!'
        assert sType in [ 'EQUITY', 'OPTION' ], 'Incorrect sType!'

        if sType == 'EQUITY':
            assert action in [ 'BUY',
                               'SELL',
                               'SELL_SHORT' ], 'Incorrect action!'
        elif sType == 'OPTION': 
            assert action in [ 'BUY_TO_OPEN',
                               'SELL_TO_CLOSE',
                               'BUY_TO_CLOSE',
                               'SELL_TO_OPEN' ], 'Incorrect action!'

        orderHash = {
            'orderType': orderType,
            'duration': duration,
            'session': 'NORMAL',
            'orderStrategyType': 'SINGLE',
            'orderLegCollection': [
                {
                    'instruction': action,
                    'quantity': quantity,
                    'instrument': {
                        'symbol': symbol,
                        'assetType': sType
                    }
                }
            ]
        }

        if orderType == 'LIMIT':
            orderHash[ 'price' ] = price

        url = 'https://api.tdameritrade.com/v1/accounts/%s/orders' % \
            self.accountId
        
        headers = { 'Authorization': 'Bearer ' + self.token }
        
        resp = requests.post( url,
                              headers = headers,
                              json    = orderHash    )

        if resp.status_code != 200:
            self.logger.error( resp.text )
        else:
            self.logger.info( resp.text )
