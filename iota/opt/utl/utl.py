import logging
import time
from functools import wraps
import pandas as pd
import yfinance as yf

def get_logger(
    pkg_name, 
    log_file_name=None,
    verbose=1,
):
    
    verbose_hash = {
        0: logging.CRITICAL,
        1 :logging.INFO,
        2 :logging.DEBUG
    }
        
    logger = logging.getLogger(pkg_name)

    logger.handlers = []
    
    logger.setLevel(verbose_hash[verbose])
        
    if log_file_name is None:
        f_hd = logging.StreamHandler() 
    else:
        f_hd = logging.FileHandler(log_file_name)

    log_fmt = logging.Formatter(
        "%(asctime)s - %(name)s %(levelname)-s - %(message)s"
    )
        
    f_hd.setFormatter(log_fmt)
        
    logger.addHandler(f_hd)

    return logger


def timer(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger("time")

        while len(logger.handlers) > 1:
            logger.handlers.pop()

        beg_time = time.time()
        val = func(*args, **kwargs)
        end_time = time.time()

        tot_time = end_time - beg_time

        logger.info(
            f"Runtime of %s: %0.1f seconds!" % (
                func.__name__,
                tot_time,
            )
        )

        return val

    return wrapper


def get_price(symbol: str):

    try:
        df = yf.Ticker(symbol).history(period="max", interval="1d")
        df = df[["Close"]]
        df["Date"] = df.index
        df = df.reset_index(drop=True)        
        df = df.sort_values("Date", ascending=True)
        df = df.reset_index(drop=True)        
        df[symbol] = df["Close"]
        df = df[["Date", symbol]]
    except Exception as exc:
        print(exc)
        return None

    return df

def get_prices(symbols: list):

    df = None
    for symbol in symbols:
      tmp_df = get_price(symbol)
      
      if tmp_df is None or tmp_df.dropna().shape[0] == 0:
          print("Symbol <%s> not found!" % symbol)
      else:
          if df is None:
              df = tmp_df
          else:
              df = df.merge(tmp_df, on="Date", how="outer")

    df = df.sort_values("Date", ascending=True)
    
    return df
              
def get_returns(symbols):

    df = get_prices(symbols)
    df = df.sort_values("Date", ascending=True)
    
    symbols = list(set(symbols).intersection(set(df.columns)))
    
    for symbol in symbols:
        df[symbol] = df[symbol].pct_change()

    return df
