# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import requests
import tdameritrade
import tdameritrade.auth as auth

from collections import defaultdict

from tdameritrade import TDClient

sys.path.append( os.path.abspath( '../' ) )

from utl.utils import getLogger

# ***********************************************************************
# Some parameters
# ***********************************************************************

MAX_SLIP = 2.0
TOL_SLIP = 0.0025

MAX_RETRIES = 10

ORDER_WAIT_TIME = 10
RETRY_WAIT_TIME = 20

# ***********************************************************************
# Class Tdam: A class to trade with TD Ameritrade
# ***********************************************************************

class Tdam:

    def __init__(   self,
                    authKey      = None,
                    refToken     = None,
                    callbackUrl  = 'http://localhost:8080',
                    accountId    = None,
                    maxSlip      = MAX_SLIP,
                    tolSlip      = TOL_SLIP,                    
                    logFileName  = None,                    
                    verbose      = 1          ):

        if authKey is None:
            self.authKey  = 'QKSFXGSPRYAV3FMQMKTA23WXSDGSJL2A'
        else:
            self.authKey  = authKey

        self.refToken    = refToken

        self.callbackUrl = callbackUrl
        self.maxSlip     = maxSlip
        self.tolSlip     = tolSlip        
        self.logFileName = logFileName
        self.verbose     = verbose
        self.logger      = getLogger( logFileName, verbose, 'tdam' )
        
        self.token       = None
        self.client      = None
        self.accounts    = None
        self.accountId   = None
        self.accessHash  = None

        for itr in range( MAX_RETRIES ):
            
            try:
                self.setAuth()
                self.setAccounts()
                self.setAccountId( accountId )
                break
            except Exception as e:
                if itr == MAX_RETRIES - 1:
                    self.logging.error( 'Failed to authenticate!' )
                else:
                    self.logger.warning( '%s: Retrying in %d seconds!',
                                         e,
                                         RETRY_WAIT_TIME )
                    time.sleep( RETRY_WAIT_TIME )            

    def setAuth( self ):

        if self.refToken is None:
            accessHash = auth.authentication( client_id    = self.authKey,
                                              redirect_uri = self.callbackUrl )
            self.refToken = accessHash[ 'refresh_token' ]
        else:
            accessHash = auth.refresh_token( refresh_token = self.refToken,
                                             client_id     = self.authKey     )

        self.token      = accessHash[ 'access_token' ]
        self.client     = TDClient( self.token )
        self.accessHash = accessHash
        
    def setAccounts( self ):

        self.accounts = self.client.accounts( positions = True,
                                              orders    = True )

        assert len( self.accounts.keys() ) > 0, 'No accounts found!'

    def setAccountId( self, accountId = None ):

        if accountId is None:
            self.accountId = list( self.accounts.keys() )[0]
        else:
            self.accountId = str( accountId )

    def getTotalValue( self ):
        
        account = self.accounts[ self.accountId ][ 'securitiesAccount' ]
        totVal  = account['currentBalances']['liquidationValue']

        return totVal
        
    def getCashBalance( self ):

        account = self.accounts[ self.accountId ][ 'securitiesAccount' ]
        cashBal = account['currentBalances']['cashBalance'] + \
                  account['currentBalances']['moneyMarketFund']

        return cashBal

    def getOpenOrders( self ):

        account = self.accounts[ self.accountId ][ 'securitiesAccount' ]

        if 'orderStrategies' in account.keys():
            val = account[ 'orderStrategies' ]
        else:
            val = None

        return val

    def getPositions( self ):

        account = self.accounts[ self.accountId ][ 'securitiesAccount' ]

        if 'positions' not in account.keys():
            positions = []
        else:
            positions =  account[ 'positions' ]

        if positions is None:
            positions = []

        return positions
    
    def getQuote( self, symbol, pType = 'last' ):

        if pType == 'ask':
            item = 'askPrice'
        elif pType == 'bid':
            item = 'bidPrice'
        else:
            item = 'lastPrice'

        for itr in range( MAX_RETRIES ):
            
            try:
                price = self.client.quote( symbol )[ symbol ][ item ]
                break
            except Exception as e:
                if itr == MAX_RETRIES - 1:
                    self.logger.error( 'Failed to get a quote for %s!',
                                       symbol )
                else:
                    self.logger.warning( '%s: Retrying in %d seconds!',
                                         e,
                                         RETRY_WAIT_TIME )
                    time.sleep( RETRY_WAIT_TIME )            

        return price

    def getOptionsChain( self, symbol ):

        for itr in range( MAX_RETRIES ):
            
            try:
                topHash = self.client.options( symbol )
                break
            except Exception as e:
                if itr == MAX_RETRIES - 1:
                    self.logger.error( 'Failed to get options chain for %s!',
                                       symbol )
                else:
                    self.logger.warning( '%s: Retrying in %d seconds!',
                                         e,
                                         RETRY_WAIT_TIME )
                    time.sleep( RETRY_WAIT_TIME )            
                
        callHash  = topHash[ 'callExpDateMap' ]
        putHash   = topHash[ 'putExpDateMap' ]
        options   = []

        for date in callHash:
            
            stkHash  = callHash[ date ]
            exprDate = date.split( ':' )[0]
            
            for strike in stkHash:
                
                obj = stkHash[ strike ][0]
                
                if symbol != obj[ 'symbol' ].split( '_' )[0]:
                    self.logger.warning ( 'Skipping: inconsistent underlying symbol %s vs. %s!',
                                          obj[ 'symbol' ],
                                          symbol,
                                          obj[ 'symbol' ].split( '_' )[0] )
                    continue
                if float( strike ) != float( obj[ 'strikePrice' ] ):
                    self.logger.warning( 'Skipping %s: inconsistent strikePrices %s vs. %s!',
                                         obj[ 'symbol' ],
                                         str( strike ),
                                         str( obj[ 'strikePrice' ] ) )
                    continue

                option = { 'optionSymbol' : obj[ 'symbol' ],
                           'assetSymbol'  : symbol,
                           'strike'       : strike,
                           'expiration'   : exprDate,
                           'contractCnt'  : obj[ 'multiplier' ],                     
                           'unitPriceAsk' : obj[ 'ask' ],
                           'unitPriceBid' : obj[ 'bid' ],
                           'unitPriceLast': obj[ 'last' ],                           
                           'type'         : 'call'      }
                
                options.append( option )

        for date in putHash:
            
            stkHash  = putHash[ date ]
            exprDate = date.split( ':' )[0]
            
            for strike in stkHash:
                
                obj = stkHash[ strike ][0]

                if symbol != obj[ 'symbol' ].split( '_' )[0]:
                    self.logger.warning( 'Skipping: inconsistent underlying symbol %s vs. %s!',
                                         obj[ 'symbol' ],
                                         symbol,
                                         obj[ 'symbol' ].split( '_' )[0] )
                    continue
                if float( strike ) != float( obj[ 'strikePrice' ] ):
                    self.logger.warning( 'Skipping %s: inconsistent strikePrices %s vs. %s!',
                                         obj[ 'symbol' ],
                                         str( strike ),
                                         str( obj[ 'strikePrice' ] ) )
                    continue                
                
                option = { 'optionSymbol' : obj[ 'symbol' ],
                           'assetSymbol'  : symbol,
                           'strike'       : strike,
                           'expiration'   : exprDate,
                           'contractCnt'  : obj[ 'multiplier' ],                     
                           'unitPriceAsk' : obj[ 'ask' ],
                           'unitPriceBid' : obj[ 'bid' ],
                           'unitPriceLast': obj[ 'last' ],                           
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

        self.logger.info( '%s %d %s...', action, quantity, symbol )
        
        assert duration in  [ 'DAY',
                              'GOOD_TILL_CANCEL',
                              'FILL_OR_KILL' ], 'Incorrect duration!'
        
        assert sType in [ 'EQUITY', 'OPTION' ], 'Incorrect sType!'

        if sType == 'EQUITY':
            assert orderType in [ 'MARKET',
                                  'LIMIT' ], 'Incorrect orderType!'            
            assert action in [ 'BUY',
                               'SELL',
                               'SELL_SHORT' ], 'Incorrect action!'
        elif sType == 'OPTION':
            assert orderType in [ 'MARKET',
                                  'LIMIT',
                                  'EXERCISE' ], 'Incorrect orderType!'            
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

        for itr in range( MAX_RETRIES ):
            resp = requests.post( url,
                                  headers = headers,
                                  json    = orderHash    )
            if resp.ok:
                break
            else:
                self.logger.warning( '%s: Retrying in %d seconds!', resp.text, RETRY_WAIT_TIME )
                time.sleep( RETRY_WAIT_TIME )

        if not resp.ok:
            self.logger.critical( 'Failed to trade %s: %s', symbol, resp.text )
                
        self.logger.info( resp.text )
            
    def getPortfolio( self, sType = 'EQUITY' ):
        
        positions = self.getPositions()
        prtHash   = defaultdict( int )
        
        for position in positions:

            if sType != position[ 'instrument' ][ 'assetType' ]:
                continue

            symbol   = position[ 'instrument' ][ 'symbol' ]
            longQty  = position[ 'longQuantity' ]
            shortQty = position[ 'shortQuantity' ]
            quantity = abs( longQty ) - abs( shortQty )
            
            prtHash[ symbol ] += quantity

        
        return prtHash

    def setPortfolio( self, orderQtyHash, sType = 'EQUITY' ):

        orderList = list( orderQtyHash.keys() )
        orderList = sorted( orderList,
                            key = lambda x: orderQtyHash[ x ] )
        
        for symbol in orderList:
            
            currPrice = self.getQuote( symbol )
            quantity  = orderQtyHash[ symbol ]

            if quantity > 0:
                orderAction = 'BUY'
            elif quantity < 0:
                orderAction = 'SELL'
                quantity = -quantity
            else:
                continue

            try:
                self.order( symbol,
                            orderType = 'MARKET',
                            duration  = 'DAY',
                            price     = None,
                            quantity  = quantity,
                            sType     = sType,
                            action    = orderAction )

                time.sleep( ORDER_WAIT_TIME )
            except Exception as e:
                self.logger.critical( e )

    def adjSlip( self, orderQtyHash, currPrtHash ):

        for symbol in orderQtyHash:
            
            qty  = orderQtyHash[ symbol ]
            last = self.getQuote( symbol, 'last' )

            lastInv = 0.0
            if last > 0:
                lastInv = 1.0 / last
            else:
                self.logger.error( 'Not trading %s as encountered non-postive '
                                   'last price of %0.2f!',
                                   symbol,
                                   last )
                orderQtyHash[ symbol ] = 0
                continue                

            if qty >= 0:
                actual = self.getQuote( symbol, 'ask' )
            elif qty < 0:
                actual = self.getQuote( symbol, 'bid' )

            if actual <= 0:
                self.logger.error( 'Not trading %s as encountered non-postive '
                                   'ask or bid price of %0.2f!',
                                   symbol,
                                   actual )
                orderQtyHash[ symbol ] = 0
                continue
            
            slip = abs( actual - last ) * lastInv
            
            if slip > self.maxSlip:
                self.logger.critical( 'Not trading %s as slip of %0.3f is '
                                      'larger than the threshold!',
                                      symbol,
                                      slip )
                orderQtyHash[ symbol ] = 0
                continue

            elif slip > self.tolSlip and qty > 0:
                adjQty = int( qty * last / actual )

                if adjQty < 0 and currPrtHash[ symbol ] > 0:
                    adjQty = -min( abs( adjQty ), currPrtHash[ symbol ] )
                    
                orderQtyHash[ symbol ] = adjQty
                
                self.logger.info( 'Adjusting order quantity of %s from %d to %d!',
                                  symbol,
                                  qty,
                                  adjQty )

        return orderQtyHash
            
    def adjWeights( self, wtHash, invHash, totVal = None ):

        # Echo some info

        tmpStr = ''
        for asset in wtHash:
            perc    = 100.0 * wtHash[ asset ]
            tmpStr += '%10s: %0.2f %s\n' % ( asset, perc, '%' )

        self.logger.info( 'Implementing the following portfolio weights: \n %s',
                          tmpStr )
        
        # Check sanity of the inverse ETF hash
        
        for symbol in wtHash:
            assert symbol in invHash.keys(),\
                'Error: no inverse found for %s!' % symbol

            assert invHash[ symbol ] not in wtHash.keys(),\
                'Error: weight hash should not contain both asset %s and its inverse %s!' \
                % ( symbol, invHash[ symbol ] )

        # Normalize weight hash
        
        totWt = 0.0
        for symbol in wtHash:
            totWt += abs( wtHash[ symbol ] )

        totWtInv = totWt
        if totWtInv > 0:
            totWtInv = 1.0 / totWtInv
            
        for symbol in wtHash:
            wtHash[ symbol ] = wtHash[ symbol ] * totWtInv

        # Get total value
        
        totValAll = self.getTotalValue()

        if totVal is None:
            totVal = totValAll
        else:
            assert totVal <= totValAll, \
                'Total value specified cannot be larger than total value of account!'

        assert totVal > 0, 'Total value should be positive'

        # Get current portfolio
        
        currPrtHash  = self.getPortfolio()

        # Initialize order hash
        
        orderQtyHash = defaultdict( int )        

        # Close any postions of irrelevant symbols

        symbols = []
        for symbol in wtHash:
            symbols.append( symbol )
            symbols.append( invHash[ symbol ] )

        for symbol in currPrtHash:
            
            if symbol in symbols:
                continue

            orderQtyHash[ symbol ] += -currPrtHash[ symbol ]

        # Adjust to meet target weights
        
        for symbol in wtHash:

            invSymbol  = invHash[ symbol ]
            currQty    = currPrtHash[ symbol ]
            currInvQty = currPrtHash[ invSymbol ]

            if currQty * currInvQty != 0:
                self.logger.warning( '%d of %s and %d of its inverse %s are in current portfolio!',
                                     currQty,
                                     symbol,
                                     currInvQty,
                                     invSymbol ) 
        
            targWt = wtHash[ symbol ]

            if targWt >= 0:

                if currInvQty != 0:
                    orderQtyHash[ invSymbol ] += -currInvQty
                
                price = self.getQuote( symbol )
                
                assert price > 0, 'Price of %s should be positive' % symbol

                targQty = int( targWt * totVal / price )

                assert targQty >= 0, 'Target quantity for symbol shoul;d be positive!' \
                    % symbol
                
                orderQtyHash[ symbol ] += targQty - currQty
                
            elif targWt < 0:

                if currQty != 0:
                    orderQtyHash[ symbol ] += -currQty

                invPrice = self.getQuote( invSymbol )
                
                assert invPrice > 0, 'Price of %s should be positive' % invSymbol

                targInvQty = int( -targWt * totVal / invPrice )

                assert targInvQty > 0, 'Taget quantity for inverse symbol %s should be positive!' % \
                    invSymbol
                
                orderQtyHash[ invSymbol ] += targInvQty - currInvQty

        # Adjust for ask/last or bid/last slip

        orderQtyHash = self.adjSlip( orderQtyHash, currPrtHash )
        
        # Implement the orders
        
        self.logger.info( str(orderQtyHash) )

        self.setPortfolio( orderQtyHash )    

    def orderVos( self, pairHash, quantity ):

        self.logger.info(   'Ordering %d of %s/%s...',
                            quantity,
                            pairHash[ 'optionSymbolBuy' ],
                            pairHash[ 'optionSymbolSell' ]
                         )

        optionSymbolBuy  = pairHash[ 'optionSymbolBuy' ]
        optionSymbolSell = pairHash[ 'optionSymbolSell' ]        

        orderHash = {
            'orderType': 'MARKET',
            'session': 'NORMAL',
            'duration': 'DAY',
            'orderStrategyType': 'SINGLE',
            'complexOrderStrategyType': 'VERTICAL',
            'orderLegCollection': [
                {
                    'instruction': 'BUY_TO_OPEN',
                    'quantity': quantity,
                    'instrument': {
                        'symbol': optionSymbolBuy,
                        'assetType': 'OPTION'
                    }
                },
                {
                    'instruction': 'SELL_TO_OPEN',
                    'quantity': quantity,
                    'instrument': {
                        'symbol': optionSymbolSell,
                        'assetType': 'OPTION'
                    }
                }
            ]
        }        

        url = 'https://api.tdameritrade.com/v1/accounts/%s/orders' % \
            self.accountId
        
        headers = { 'Authorization': 'Bearer ' + self.token }

        for itr in range( MAX_RETRIES ):
            resp = requests.post( url,
                                  headers = headers,
                                  json    = orderHash )
            if resp.ok:
                break
            else:
                self.logger.warning(
                    '%s: Retrying in %d seconds!',
                    resp.text,
                    RETRY_WAIT_TIME
                )
                time.sleep( RETRY_WAIT_TIME )

        if not resp.ok:
            self.logger.critical(
                'Failed to trade %s/%s: %s',
                pairHash[ 'optionSymbolBuy' ],
                pairHash[ 'optionSymbolSell' ],
                resp.text
            )
                
        self.logger.info( resp.text )
        
