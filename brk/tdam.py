# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
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
                    callbackUrl  = 'http://localhost:8080',
                    logFileName  = None,                    
                    verbose      = 1          ):

        if authKey is None:
            self.authKey = 'QKSFXGSPRYAV3FMQMKTA23WXSDGSJL2A'
        else:
            self.authKey = authKey

        self.callbackUrl = callbackUrl
        self.logFileName = logFileName
        self.verbose     = verbose
        self.logger      = getLogger( logFileName, verbose, 'tdam' )
        
        self.token       = None
        self.client      = None
        self.accounts    = None
        self.accountId   = None

        self.setAuth()
        self.setAccount()
        self.setAccountId()

    def setAuth( self ):

        accessHash = auth.authentication( client_id    = self.authKey,
                                          redirect_uri = self.callbackUrl )        
        
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
        
    def getQuote( self, symbol ):

        price = self.client.quote( symbol )[ symbol ][ 'lastPrice' ]

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
                    'Inconsistent underlying symbol!'

                assert strike == obj[ 'strikePrice' ], \
                    'Inconsistent strikePrice!'
                
                option = { 'optionSymbol' : obj[ 'symbol' ],
                           'assetSymbol'  : symbol,
                           'strike'       : strike,
                           'expiration'   : exprDate,
                           'contractCnt'  : obj[ 'multiplier' ],                     
                           'unitPrice'    : obj[ 'last' ],
                           'type'         : 'call'      }
                
                options.append( option )

        for date in putHash:
            
            stkHash  = putHash[ date ]
            exprDate = date.split( ':' )[0]
            
            for strike in stkHash:
                
                obj = stkHash[ strike ][0]

                assert symbol == obj[ 'symbol' ].split( '_' )[0], \
                    'Inconsistent underlying symbol!'
                
                assert strike == obj[ 'strikePrice' ], \
                    'Inconsistent strikePrice!'
                
                option = { 'optionSymbol' : obj[ 'symbol' ],
                           'assetSymbol'  : symbol,
                           'strike'       : strike,
                           'expiration'   : exprDate,
                           'contractCnt'  : obj[ 'multiplier' ],                     
                           'unitPrice'    : obj[ 'last' ],
                           'type'         : 'put'      }
                
                options.append( option )
                
        return options

    def order( self, symbol, orderType = 'MARKET', sType = 'EQUITY' ):

        url = 'https://api.tdameritrade.com/v1/accounts/%s/orders' % \
            self.accountId

        orderHash = {}
        
