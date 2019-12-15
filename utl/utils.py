# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import datetime
import requests
import logging
import numpy as np
import pandas as pd

from logging.handlers import SMTPHandler

# ***********************************************************************
# getLogger(): Get a logger object
# ***********************************************************************

def getLogger( logFileName, verbose, pkgName = None ):
    
    verboseHash = { 0 : logging.NOTSET,
                    1 : logging.INFO,
                    2 : logging.DEBUG }
        
    logger      = logging.getLogger( pkgName )
        
    logger.setLevel( verboseHash[ verbose ] )
        
    if logFileName is None:
        fHd = logging.StreamHandler() 
    else:
        fHd = logging.FileHandler( logFileName )

    logFmt = logging.Formatter( '%(asctime)s - %(name)s %(levelname)-s - %(message)s' )
        
    fHd.setFormatter( logFmt )
        
    logger.addHandler( fHd )

    return logger

# ***********************************************************************
# getLogger(): Get a logger object
# ***********************************************************************

def getAlertHandler( alertLevel, subject = None, mailList = [] ):
    
    mHd = SMTPHandler( mailhost    = ( 'smtp.gmail.com', 587 ),
                       fromaddr    = 'optilive.noreply@gmail.com',
                       toaddrs     = mailList,
                       subject     = subject,
                       credentials = ('optilive.noreply@gmail.com',
                                      'optilivenoreply'),
                       secure      = () )
    
    mHd.setLevel( alertLevel )
        
    return mHd
