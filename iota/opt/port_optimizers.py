import os
import sys

sys.path.append("../..")

import numpy as np
import pandas as pd
import scipy as sp

from utl.utl import get_logger, timer

class MomentPortOpt:
    def __init__(
        self,
        return_df: pd.DataFrame,
        asset_universe: list,            
        num_lookback_days: int,
        num_sub_win_days: int,
        num_olap_days: int,
        min_num_sub_win_days: int,
        logger=None,            
    ):
        super(MomentPortOpt, self).__init__()

        for asset in asset_universe:
            assert asset in return_df.columns, "Asset %s not found!" % asset
            return_df[asset] = return_df[asset].fillna(method="ffill").fillna(method="bfill")
            
        self.return_df = return_df
        self.num_lookback_days = num_lookback_days
        self.num_sub_win_days = num_sub_win_days
        self.num_olap_days = num_olap_days
        self.min_num_sub_win_days = min_num_sub_win_days        
        self.asset_universe = list(asset_universe)

        for item in self.asset_universe:
            assert item in self.return_df.columns

        assert "Date" in self.return_df.columns

        self.return_df["Date"] = self.return_df["Date"].astype("datetime64[ns]")
        
        if logger is None:
            self.logger = get_logger("portfolio_optimizor")
        else:
            self.logger = logger

    @timer
    def get_optimal_allocation(self, snap_date=None):
        
        if snap_date is None:
            snap_date = self.return_df["Date"].max()

        max_date = pd.to_datetime(snap_date)
        min_date = max_date - datetime.timedelta(days=self.num_lookback_days)

        if self.return_df["Date"].min() > min_date:
            self.logger.warning(
                "Minimum date in data %s is larger than expected minimum date",
                self.return_df["Date"].min().strftime("%Y-%m-%d"),
                min_date,
            )

        if self.return_df["Date"].max() < max_date:
            self.logger.warning(
                "Maximum date in data %s is smaller than the expected maximum date",
                self.return_df["Date"].min().strftime("%Y-%m-%d"),
                max_date,
            )
            
        num_assets = len(self.asset_universe)

        assert num_assets > 0, "No assets found!"
        
        weights = np.ones(shape=(num_assets), dtype=np.float64)
        weights /= num_assets
        
    def get_obj(self, weights, min_date, max_date):

        min_date = pd.to_datetime(min_date)
        max_date = pd.to_datetime(max_date)        
        weights = np.array(weights)

        return_df = self.return_df
        beg_date = min_date
        exp_return = 0.0
        m = 0
        while beg_date <= max_date - datetime.timedelta(days=self.min_num_sub_win_days):
            
            end_date = beg_date + datetime.timedelta(days=self.num_sub_win_days)
            end_date = min(end_date, max_date)
            
            tmp_df = return_df[
                (return_df["Date"] >= beg_date) &
                (return_df["Date"] <= end_date)
            ]
            for i, asset in enumerate(self.asset_universe):
                exp_return += weights[i] * tmp_df[asset].mean()
                
            m += 1
            beg_date += datetime.timedelta(days=self.num_sub_win_days - self.num_olap_days)
            
        fct = m
        if fct > 0:
            fct = 1.0 / fct

        exp_return *= fct

        return -exp_return


    def get_sub_win_variances(self, weights, min_date, max_date):

        min_date = pd.to_datetime(min_date)
        max_date = pd.to_datetime(max_date)        
        weights = np.array(weights)

        num_assets = len(self.asset_universe)

        assert len(weights) == num_assets, "Inconsistent weights size!"
        
        return_df = self.return_df
        beg_date = min_date
        variances = []
        while beg_date <= max_date - datetime.timedelta(days=self.min_num_sub_win_days):
            
            end_date = beg_date + datetime.timedelta(days=self.num_sub_win_days)
            end_date = min(end_date, max_date)
            
            tmp_df = return_df[
                (return_df["Date"] >= beg_date) &
                (return_df["Date"] <= end_date)
            ]

            r_list = []
            for asset in enumerate(self.asset_universe):
                r_list.append(np.array(tmp_df[asset]))

            cov_mat = np.cov(r_list)

            variances.append(weights @ cov_mat @ weights)
            
            beg_date += datetime.timedelta(days=self.num_sub_win_days - self.num_olap_days)
            
        return variances
    
