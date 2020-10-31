# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import robin_stocks as rs

from collections import defaultdict

sys.path.append( os.path.abspath( '../' ) )

from utl.utils import getLogger

# ***********************************************************************
# Some parameters
# ***********************************************************************

MAX_SLIP = 2.0
TOL_SLIP = 0.0025

MAX_RETRIES = 5

ORDER_WAIT_TIME = 10
RETRY_WAIT_TIME = 20

# ***********************************************************************
# Class Tdam: A class to trade with TD Ameritrade
# ***********************************************************************

class Rbin:

    def __init__(   self,
                    userName    = None,
                    PassKey     = None,
                    maxSlip      = MAX_SLIP,
                    tolSlip      = TOL_SLIP,                    
                    logFileName  = None,                    
                    verbose      = 1          ):

        self.maxSlip     = maxSlip
        self.tolSlip     = tolSlip        
        self.logFileName = logFileName
        self.verbose     = verbose
        self.logger      = getLogger( logFileName, verbose, 'tdam' )
        
        for itr in range( MAX_RETRIES ):
            
            try:
                self.setAuth( userName, passKey )
                break
            except Exception as e:
                if itr == MAX_RETRIES - 1:
                    self.logging.error( 'Failed to authenticate!' )
                else:
                    self.logger.warning( '%s: Retrying in %d seconds!',
                                         e,
                                         RETRY_WAIT_TIME )
                    time.sleep( RETRY_WAIT_TIME )            

    def setAuth( self, userName, passKey ):

        authHash = rs.login( username  = userName,
                             password  = passKey,
                             expiresIn = 86400,
                             by_sms    = False      )
        
        if authHash is None:
            self.logger.error( 'Failed to authenticate Robinhood!' )
            
    def unSetAuth( self ):

        rs.logout()

    def getQuote( self, symbol, pType ):

        quoteHash = rs.crypto.get_crypto_quote( symbol )

        price = None
        
        if pType == 'ask':
            price = quoteHash[ 'ask_price' ]
        elif pType == 'bid':
            price = quoteHash[ 'bid_price' ]
        else:
            price = quoteHash[ 'mark_price' ]

        return price
        
    def getCashBalance( self ):

        tmpHash = rs.profiles.load_account_profile()

        cashBal = None
        
        if tmpHash is not None and 'cash' in tmpHash:
            cashBal = tmphash[ 'cash' ]
        else:
            self.logger.error( 'Could not get cash balance!' )
            
        return cashBal

    def getTotalValue( self ):

        totVal = self.getCashBalance()

        tmpHash = rs.account.build_holdings()

        for symbol in tmpHash:
            price   = tmpHash[ symbol ][ 'price' ]
            qty     = tmpHash[ symbol ][ 'quantity' ]
            totVal += price * qty

        return totVal

    def getPortfolio( self ):

        tmpHash = rs.account.build_holdings()
        
        prtHash = {}
        for symbol in tmpHash:
            prtHash[ symbol ] = tmpHash[ symbol ][ 'quantity' ]
            
        return prtHash
    
    def setPortfolio( self, orderQtyHash ):

        orderList = list( orderQtyHash.keys() )
        orderList = sorted( orderList,
                            key = lambda x: orderQtyHash[ x ] )
        
        for symbol in orderList:
            
            quantity  = orderQtyHash[ symbol ]

            for itr in range( MAX_RETRIES ):
                
                try:
                    if quantity > 0:
                        rs.orders.order_buy_crypto_by_quantity(
                            symbol,
                            quantity
                        )
                    elif quantity < 0:
                        rs.orders.order_sell_crypto_by_quantity(
                            symbol,
                            abs( quantity )
                        )
                except Exception as e:
                    if itr == MAX_RETRIES - 1:
                        self.logging.error( 'Failed to trade %s!', symbol )
                    else:
                        self.logger.warning( '%s: Retrying in %d seconds!',
                                             e,
                                             RETRY_WAIT_TIME )
                        time.sleep( RETRY_WAIT_TIME )            

            time.sleep( ORDER_WAIT_TIME )

    def adjWeights( self, wtHash, totVal = None ):

        # Echo some info

        tmpStr = ''
        for asset in wtHash:
            perc    = 100.0 * wtHash[ asset ]
            tmpStr += '%10s: %0.2f %s\n' % ( asset, perc, '%' )

        self.logger.info( 'Implementing the following portfolio weights: \n %s',
                          tmpStr )
        
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
        
        orderQtyHash = defaultdict( float )        

        # Close any postions of irrelevant symbols

        for symbol in currPrtHash:
            
            if symbol in wtHash.keys():
                continue

            orderQtyHash[ symbol ] += -currPrtHash[ symbol ]

        # Adjust to meet target weights
        
        for symbol in wtHash:

            targWt  = wtHash[ symbol ]

            assert targWt >= 0, 'Only non-negative weights are supported at this time!'

            currQty = currPrtHash[ symbol ]
            currWt  = currQty / totVal

            if targWt > currWt:
                price = self.getQuote( symbol, 'ask' )
            elif targWt < currWt:
                price = self.getQuote( symbol, 'bid' )
            else:
                continue
            
            assert price > 0, 'Price of %s should be positive' % symbol
            
            ordQty = ( targWt - currWt ) * totVal / price

            orderQtyHash[ symbol ] += orderQty
                
        # Implement the orders
        
        self.logger.info( str(orderQtyHash) )

        self.setPortfolio( orderQtyHash )    
