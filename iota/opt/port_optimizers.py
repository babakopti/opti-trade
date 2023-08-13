import os
import sys

sys.path.append("..")

import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import minimize

from utl.utl import get_logger, timer


class MomentPortOpt:
    def __init__(
        self,
        daily_return_df: pd.DataFrame,
        asset_universe: list,
        num_lookback_days: int,
        num_sub_win_days: int,
        num_olap_days: int,
        min_num_sub_win_days: int,
        variance_limit_alpha: float = 1.2,
        variance_limit_indicator: str = "SPY",
        variance_limit_mode: str = "average",
        max_asset_weight: float = 0.10,
        logger=None,
    ):
        super(MomentPortOpt, self).__init__()

        for asset in list(asset_universe) + [variance_limit_indicator]:
            assert asset in daily_return_df.columns, "Asset %s not found!" % asset
            daily_return_df[asset] = (
                daily_return_df[asset].fillna(method="ffill").fillna(method="bfill")
            )

        self.return_df = daily_return_df
        self.num_lookback_days = num_lookback_days
        self.num_sub_win_days = num_sub_win_days
        self.num_olap_days = num_olap_days
        self.min_num_sub_win_days = min_num_sub_win_days
        self.asset_universe = list(asset_universe)

        for item in self.asset_universe:
            assert item in self.return_df.columns

        assert "Date" in self.return_df.columns

        self.return_df["Date"] = self.return_df["Date"].astype("datetime64[ns]")

        self.variance_limit_alpha = variance_limit_alpha
        self.variance_limit_indicator = variance_limit_indicator
        self.variance_limit_mode = variance_limit_mode
        self.max_asset_weight = max_asset_weight
        
        assert variance_limit_mode in ["average", "all"], "Mode not known!"

        if logger is None:
            self.logger = get_logger("portfolio_optimizor")
        else:
            self.logger = logger

        self.port_variances = None
        self.port_variance_jacs = None
        self.ind_variances = None

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

        init_weights = np.ones(shape=(num_assets), dtype=np.float64)
        init_weights /= num_assets

        constraints = self._get_constraints(min_date, max_date)

        results = minimize(
            fun=lambda x: self._get_exp_return(x, min_date, max_date)
            x0=init_weights,
            method="SLSQP",
            tol=1.0e-6,
            constraints=constraints,
            bounds=[(0.0, self.max_asset_weight)] * num_assets,
            options={"maxiter": 10000},
        )

        self.logger.info(results["message"])
        self.logger.info("Optimization success: %s", str(results["success"]))
        self.logger.info("Number of function evals: %d", results["nfev"])

        weights = results.x

        assert len(weights) == num_assets, "Inconsistent size of weights!"

        weight_hash = {}
        for i, asset in enumerate(self.asset_universe):
            if weights[i] > 0:
                weight_hash[asset] = weights[i]

        return weight_hash
    
    def _get_exp_return(self, weights, min_date, max_date):

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
                (return_df["Date"] >= beg_date) & (return_df["Date"] <= end_date)
            ]
            for i, asset in enumerate(self.asset_universe):
                exp_return += weights[i] * tmp_df[asset].mean()

            m += 1
            beg_date += datetime.timedelta(
                days=self.num_sub_win_days - self.num_olap_days
            )

        fct = m
        if fct > 0:
            fct = 1.0 / fct

        exp_return *= fct

        return -exp_return

    def _get_constraints(self, min_date, max_date):

        sub_win_dates = self._get_sub_win_dates(min_date, max_date)

        num_sub_wins = len(sum_win_dates)

        assert len(num_sub_wins) > 0, "Internal error!"

        funcs = []
        jacs = []
        alpha = self.variance_limit_alpha
        for m in range(len(sub_win_dates)):
            funcs.append(
                lambda x: alpha ** 2 * self._get_ind_variance(m, sub_win_dates)
                - self._get_port_variance(x, m, sub_win_dates)
            )
            jacs.append(lambda x: -self._get_port_variance_jac(x, m, sub_win_dates))

        if self.variance_limit_mode == "average":
            func = lambda x: np.mean([funcs[m](x) for m in range(num_sub_wins)])
            jac = (
                lambda x: -sum([jacs[m](x) for m in range(num_sub_wins)]) / num_sub_wins
            )
            constratints = [
                {
                    "type": "ineq",
                    "func": func,
                    "jac": jac,
                }
            ]
        elif self.variance_limit_mode == "all":
            constratints = []
            for m in range(num_sub_wins):
                constratints.append(
                    {
                        "type": "ineq",
                        "func": funcs[m],
                        "jac": jacs[m],
                    }
                )
        else:
            assert False, "Mode not known!"

        return constratints

    def _get_sub_win_dates(self, min_date, max_date):

        min_date = pd.to_datetime(min_date)
        max_date = pd.to_datetime(max_date)

        sub_win_dates = []
        beg_date = min_date
        while beg_date <= max_date - datetime.timedelta(days=self.min_num_sub_win_days):

            end_date = beg_date + datetime.timedelta(days=self.num_sub_win_days)
            end_date = min(end_date, max_date)

            sub_win_dates.append((beg_date, end_date))

            beg_date += datetime.timedelta(
                days=self.num_sub_win_days - self.num_olap_days
            )

        return sub_win_dates

    def _get_port_variance(self, weights, sub_win_index, sub_win_dates):

        beg_date = sub_win_dates[sub_win_index][0]
        end_date = sub_win_dates[sub_win_index][1]

        weights = np.array(weights)

        num_assets = len(self.asset_universe)

        assert len(weights) == num_assets, "Inconsistent weights size!"

        return_df = self.return_df

        tmp_df = return_df[
            (return_df["Date"] >= beg_date) & (return_df["Date"] <= end_date)
        ]

        r_list = []
        for asset in self.asset_universe:
            r_list.append(np.array(tmp_df[asset]))

        cov_mat = np.cov(r_list)

        port_variance = weights @ cov_mat @ weights

        return port_variance

    def _get_port_variance_jac(self, weights, sub_win_index, sub_win_dates):

        beg_date = sub_win_dates[sub_win_index][0]
        end_date = sub_win_dates[sub_win_index][1]

        weights = np.array(weights)

        num_assets = len(self.asset_universe)

        assert len(weights) == num_assets, "Inconsistent weights size!"

        return_df = self.return_df

        tmp_df = return_df[
            (return_df["Date"] >= beg_date) & (return_df["Date"] <= end_date)
        ]

        r_list = []
        for asset in self.asset_universe:
            r_list.append(np.array(tmp_df[asset]))

        cov_mat = np.cov(r_list)

        port_variance_jac = 2.0 * cov_mat @ weights

        assert len(port_variance_jac) == len(weights), "Iconsistent sizes!"

        return port_variance_jac

    def _get_ind_variance(self, sub_win_index, sub_win_dates):

        beg_date = sub_win_dates[sub_win_index][0]
        end_date = sub_win_dates[sub_win_index][1]

        return_df = self.return_df

        tmp_df = return_df[
            (return_df["Date"] >= beg_date) & (return_df["Date"] <= end_date)
        ]

        return np.var(tmp_df[self.variance_limit_indicator])
