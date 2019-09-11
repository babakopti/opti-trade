# ***********************************************************************
# Import libraries
# ***********************************************************************

import os
import sys
import webbrowser
import json
import logging
import configparser
import requests
import random
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from rauth import OAuth1Service
from market.market import Market

# ***********************************************************************
# Class: Etrade
# ***********************************************************************

class Etrade:

    def __init__( self, 
                  confFile, 
                  logFile    = 'etrade.log', 
                  sandBox    = True,
                  accountId  = 0,
                  verbose    = 1              ):
        
        self.config = configparser.ConfigParser()
        self.config.read( confFile )
        
        if verbose == 0:
            level = logging.ERROR
        elif verbose == 1:
            level = logging.INFO
        else:
            level = logging.DEBUG
            
        logging.basicConfig( filename = logFile,
                             level    = level    )

        self.sandBox  = sandBox

        if sandBox:
            self.baseUrl  = self.config[ 'SANDBOX' ][ 'SANDBOX_BASE_URL' ]
            self.confHead = 'SANDBOX' 
        else:
            self.baseUrl  = self.config[ 'PRODUCTION' ][ 'PROD_BASE_URL' ]
            self.confHead = 'PRODUCTION'

        self.verbose  = verbose
        self.session  = None
        self.accounts = []
        self.account  = {}

        self.setSession()
        self.setAccounts()
        self.setAccount( accountId )

    def setSession( self, sessionType = 'sandbox' ):

        config           = self.config
        consumKey        = config[ self.confHead ][ 'CONSUMER_KEY' ]
        consumSec        = config[ self.confHead ][ 'CONSUMER_SECRET' ]
        reqTokenUrl      = 'https://api.etrade.com/oauth/request_token'
        accTokenUrl      = 'https://api.etrade.com/oauth/access_token'
        authUrl          = 'https://us.etrade.com/e/t/etws/authorize?key={}&token={}'
        baseUrl          = 'https://api.etrade.com'

        etrade           = OAuth1Service( name              = 'etrade',
                                          consumer_key      = consumKey,
                                          consumer_secret   = consumSec,
                                          request_token_url = reqTokenUrl,
                                          access_token_url  = accTokenUrl,
                                          authorize_url     = authUrl,
                                          base_url          = baseUrl             )

        reqToken, reqTokenSec = etrade.get_request_token( params = { 'oauth_callback' : 'oob', 
                                                                     'format'         : 'json' } )

        authUrl = etrade.authorize_url.format( etrade.consumer_key, reqToken )

        webbrowser.open( authUrl )

        code = input( 'Please accept agreement and enter text code from browser: ' )

        self.session = etrade.get_auth_session( reqToken,
                                                reqTokenSec,
                                                params = { 'oauth_verifier': code } )

        return

    def setAccounts( self ):

        url      = self.baseUrl + '/v1/accounts/list.json'
        response = self.session.get( url, header_auth = True )

        assert response.status_code == 200, 'No succesfull response!'
        assert response is not None, 'No response!'

        data     = response.json()

        assert data is not None, 'No data received!'

        assert 'AccountListResponse' in data, 'Bad data received!'
        assert 'Accounts' in data[ 'AccountListResponse' ], 'Bad data received!'
        assert 'Account' in data[ 'AccountListResponse' ][ 'Accounts' ], 'No accounts found!'

        self.accounts = data[ 'AccountListResponse' ][ 'Accounts' ][ 'Account' ]
 
        if self.verbose > 0:
            print( self.accounts )

        return True

    def setAccount( self, accountId = 0 ):

        self.account = self.accounts[accountId]

    def getQuote( self, symbols ):

        market = Market( self.session, self.baseUrl )
        
        market.quotes( symbols )
    
        return True

    def getBalance( self ):
        
        url      = self.baseUrl + '/v1/accounts/' +\
                   self.account[ 'accountIdKey' ] + '/balance.json'
        params   = { 'instType'    : self.account[ 'institutionType' ], 
                     'realTimeNAV' : 'true' }
        headers  = { 'consumerkey' : self.config[ self.confHead ][ 'CONSUMER_KEY' ] }

        response = self.session.get( url, 
                                     header_auth = True, 
                                     params      = params, 
                                     headers     = headers   )

        ret = None
        if response is not None and response.status_code == 200:
            data = response.json()
            ret  = data[ 'BalanceResponse' ]
        else:
            print( 'Balance query not successful!' )

        return ret

    def getPortfolio( self ):

        url      = self.baseUrl + '/v1/accounts/' +\
                   self.account[ 'accountIdKey' ] + '/portfolio.json'
        response = self.session.get( url, 
                                     header_auth = True   )

        retList = []
        if response is not None and response.status_code == 200:
            data     = response.json()
            retList  = data[ 'PortfolioResponse' ][ 'AccountPortfolio' ][0][ 'Position' ]
        else:
            print( 'Portfolio query not successful!' )

        return retList

    def getOpenOrders( self ):
        
        url      = self.baseUrl + '/v1/accounts/' +\
                   self.account[ 'accountIdKey' ] + '/orders.json'
        params   = { 'status' : 'OPEN' }
        headers  = { 'consumerkey': self.config[ self.confHead ][ 'CONSUMER_KEY' ] }

        response = self.session.get( url, 
                                     header_auth = True, 
                                     params      = params, 
                                     headers     = headers  )

        ret = None
        if response is not None and response.status_code == 200:
            data = response.json()
            ret  = data[ 'OrdersResponse' ][ 'Order' ]
        else:
            print( 'Open orders query not successful!' )

        return ret

    def cancel( self, orderId ):

        url     = self.baseUrl + '/v1/accounts/' +\
                  self.account[ 'accountIdKey' ] + '/orders/cancel.json'

        headers = { 'Content-Type' : 'application/xml', 
                    'consumerKey'  : self.config[ self.confHead ][ 'CONSUMER_KEY' ] }

        payload = """<CancelOrderRequest>
                         <orderId>{0}</orderId>
                    </CancelOrderRequest>
                  """

        payload  = payload.format( orderId )

        response = self.session.put( url, 
                                     header_auth = True, 
                                     headers     = headers, 
                                     data        = payload  )

        data     = response.json()

        return data

    def order( self, ordHash, action ):

        assert 'price_type' in ordHash.keys(), 'price_type not found!'
        assert 'order_term' in ordHash.keys(), 'order_term not found!'
        assert 'symbol' in ordHash.keys(), 'symbol not found!'
        assert 'order_action' in ordHash.keys(), 'order_action not found!'
        assert 'limit_price' in ordHash.keys(), 'limit_price not found!'
        assert 'quantity' in ordHash.keys(), 'quantity not found!'
        assert 'order_type' in ordHash.keys(), 'order_type not found!'
        assert 'security_type' in ordHash.keys(), 'security_type not found!'

        price_type_options    = [ 'MARKET', 'LIMIT' ]
        order_term_options    = [ 'GOOD_FOR_DAY', 'IMMEDIATE_OR_CANCEL', 'FILL_OR_KILL' ]
        order_action_options  = [ 'BUY', 'SELL', 'BUY_TO_COVER', 'SELL_SHORT' ]
        order_type_options    = [ 'EQ', 
                                  'OPTN', 
                                  'SPREADS', 
                                  'BOND', 
                                  'OPTION_EXERCISE', 
                                  'OPTION_ASSIGNMENT', 
                                  'OPTION_EXPIRED' ]
        security_type_options = [ 'EQ', 'OPTN', 'BOND', 'MF', 'MMF' ]

        assert ordHash[ 'price_type' ]in price_type_options, 'Incorrect value!'
        assert ordHash[ 'order_term' ] in order_term_options, 'Incorrect value!'
        assert ordHash[ 'order_action' ] in order_action_options, 'Incorrect value!'
        assert ordHash[ 'order_type' ] in order_type_options, 'Incorrect value!'
        assert ordHash[ 'security_type' ] in security_type_options, 'Incorrect value!'

        headers = { 'Content-Type' : 'application/xml', 
                    'consumerKey'  : self.config[ self.confHead ][ 'CONSUMER_KEY' ] }

        url     = self.baseUrl + '/v1/accounts/' +\
                  self.account[ 'accountIdKey' ] + '/orders/preview.json'

        payload = """<PreviewOrderRequest>
                         <orderType>{7}</orderType>
                         <clientOrderId>{0}</clientOrderId>
                         <Order>
                             <allOrNone>false</allOrNone>
                             <priceType>{1}</priceType>
                             <orderTerm>{2}</orderTerm>
                             <marketSession>REGULAR</marketSession>
                             <stopPrice></stopPrice>
                             <limitPrice>{3}</limitPrice>
                             <Instrument>
                                 <Product>
                                     <securityType>{8}</securityType>
                                     <symbol>{4}</symbol>
                                 </Product>
                                 <orderAction>{5}</orderAction>
                                 <quantityType>QUANTITY</quantityType>
                                 <quantity>{6}</quantity>
                             </Instrument>
                         </Order>
                     </PreviewOrderRequest>"""

        clientOrderId = str( random.randint(1000000000, 9999999999) )

        payload = payload.format( clientOrderId, 
                                  ordHash[ 'price_type' ], 
                                  ordHash[ 'order_term' ],
                                  ordHash[ 'limit_price' ], 
                                  ordHash[ 'symbol' ], 
                                  ordHash[ 'order_action' ], 
                                  ordHash[ 'quantity' ],
                                  ordHash[ 'order_type' ], 
                                  ordHash[ 'security_type' ]  )

        response = self.session.post( url, 
                                      header_auth = True, 
                                      headers     = headers, 
                                      data        = payload  )

        data      = response.json()
        data      = data[ 'PreviewOrderResponse' ]
        previewId = data[ 'PreviewIds' ][0][ 'previewId' ]

        if action != 'place':
            return data


        url     = self.baseUrl + '/v1/accounts/' +\
                  self.account[ 'accountIdKey' ] + '/orders/place.json'

        payload = """<PlaceOrderRequest>
                         <orderType>{8}</orderType>
                         <clientOrderId>{0}</clientOrderId>
                         <PreviewIds>
                             <previewId>{1}</previewId>
                         </PreviewIds>
                         <Order>
                             <allOrNone>false</allOrNone>
                             <priceType>{2}</priceType>
                             <orderTerm>{3}</orderTerm>
                             <marketSession>REGULAR</marketSession>
                             <stopPrice></stopPrice>
                             <limitPrice>{4}</limitPrice>
                             <Instrument>
                                 <Product>
                                     <securityType>{9}</securityType>
                                     <symbol>{5}</symbol>
                                 </Product>
                                 <orderAction>{6}</orderAction>
                                 <quantityType>QUANTITY</quantityType>
                                 <quantity>{7}</quantity>
                             </Instrument>
                         </Order>
                       </PlaceOrderRequest>"""

        payload = payload.format( clientOrderId,
                                  previewId,
                                  ordHash[ 'price_type' ], 
                                  ordHash[ 'order_term' ],
                                  ordHash[ 'limit_price' ], 
                                  ordHash[ 'symbol' ], 
                                  ordHash[ 'order_action' ], 
                                  ordHash[ 'quantity' ],
                                  ordHash[ 'order_type' ], 
                                  ordHash[ 'security_type' ] )

        response = self.session.post( url, 
                                      header_auth = True, 
                                      headers     = headers, 
                                      data        = payload  )

        data      = response.json()
        data      = data[ 'PlaceOrderResponse' ]

        return data

    def rebalance( self, wtHash, maxVal ):
        
        assert maxVal >= 0, 'Should be positive!'

        for symbol in wtHash:
            weight  = wtHash[ symbol ]
            price   = self.getQuote( symbol )
            qty     = int( weight * maxVal / price )

            ordHash = { 'price_type'   : 'MARKET', 
                        'order_term'   : 'GOOD_FOR_DAY', 
                        'symbol'       : symbol, 
                        'order_action' : 'BUY', 'limit_price' : 30.0, 'quantity' : 1 }

    def getMadWeights( symbols, histFile, minDate = None ):

        fileExt = histFile.split( '.' )[-1]

        if fileExt == 'csv':
            df = pd.read_csv( dfFile ) 
        elif fileExt == 'pkl':
            df = pd.read_pickle( dfFile ) 
        else:
            assert False, 'Unknown input file extension %s' % fileExt

        if minDate is not None:
            df = df[ df.Date >= pd.to_datetime( minDate ) ]

        returns   = np.log( df[ symbols ] ).pct_change().dropna()
        numAssets = len( symbols )
        guess     = np.ones( numAssets )
        cons      = { 'type' : 'eq', 'fun' : self._sum_check }

        results   = minimize( self._mad, 
                              guess, 
                              args        = returns, 
                              constraints = cons      )
    
        weights   = results.x

        assert len( weights ) == numAssest,\
            'Inconsistent size of weights!'

        wtHash    = {}
        
        for i in range( numAssets ):
            wtHash[ symbols[i] ] = weights[i]

        return wtHash

    def _sum_check( self, x ):
        return sum( abs( x ) ) - 1

    def _mad( self, x, returns ):
        return ( returns - returns.mean() ).dot( x ).abs().mean()
