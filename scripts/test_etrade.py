import os, sys

from etrade import Etrade

etrade = Etrade( 'configs/config.ini', sandBox = False )

print( etrade.getQuote( 'ALTR' ) )

print( etrade.getBalance() )

print( etrade.getPortfolio() )

print( etrade.getOpenOrders() )

etrade.cancel( orderId = 5 )

sys.exit()

ordHash = { 'price_type' : 'MARKET', 
            'order_term' : 'GOOD_FOR_DAY', 
            'symbol' : 'ALTR', 
            'order_action' : 'BUY', 
            'order_type' : 'EQ',
            'security_type' : 'EQ',
            'limit_price' : 30.0, 
            'quantity' : 1 }

print( etrade.order( ordHash, action = 'place' ) )
