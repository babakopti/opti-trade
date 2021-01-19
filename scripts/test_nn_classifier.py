# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix

# ***********************************************************************
# Define input parameters
# ***********************************************************************

FEA_FILE = "data/NNfeatures_all.csv"
RSP_FILE = "data/NNresps_all.csv"
MAX_TRN_DATE = "2020-09-30 15:30:00"
MAX_OOS_DATE = "2021-01-06 15:30:00"

RSP_VAR = "act_trend_2"
FEA_LIST = [
    "geo_trend",
    "geo_perf",
    "macd_trend",
    "ptc_tag",
]

HIDDEN_LAYER_SIZES = (3, 1)

BUILD_PRT_FLAG = True
INP_PRT_FILE = "portfolios/nTrnDays_360_two_hours_ptc.json"
OUT_PRT_FILE = "portfolios/nn_test.json"

# ***********************************************************************
# Set some parameters
# ***********************************************************************

FEA_DF = pd.read_csv(FEA_FILE)
RSP_DF = pd.read_csv(RSP_FILE)
SYMBOLS = set(FEA_DF.symbol)

NN_TREND_HASH = {}

# ***********************************************************************
# Define NN classifier function
# ***********************************************************************

def do_nn_class(symbol, hidden_layer_sizes):
    
    df = FEA_DF.merge(
        RSP_DF,
        how="inner",
        on=["Date", "symbol"],
    )

    df = df[
        (df.symbol == symbol) &
        (df.Date <= MAX_OOS_DATE) &
        (df[RSP_VAR] != 0)
    ]

    df["geo_macd_trend"] = df.apply(
        lambda x: x.geo_trend if x.geo_perf == 1 and x.geo_trend != 0\
        else x.macd_trend,
        axis=1,
    )
    df["geo_macd_ptc_trend"] = df.apply(
        lambda x: x.geo_macd_trend if x.ptc_tag == 0 \
        else x.ptc_tag,
        axis=1,
    )
    df["geo_success"] = df.apply(
        lambda x: x.geo_trend * x[RSP_VAR], axis=1
    )
    df["macd_success"] = df.apply(
        lambda x: x.macd_trend * x[RSP_VAR], axis=1
    )
    df["geo_macd_success"] = df.apply(
        lambda x: x.geo_macd_trend * x[RSP_VAR], 
        axis=1,
    )
    df["geo_macd_ptc_success"] = df.apply(
        lambda x: x.geo_macd_ptc_trend * x[RSP_VAR], 
        axis=1,
    )

    trn_df = df[df.Date <= MAX_TRN_DATE]
    oos_df = df[df.Date > MAX_TRN_DATE]

    X_trn = np.array(trn_df[FEA_LIST])
    y_trn = np.array(trn_df[RSP_VAR])
    y_trn = y_trn.reshape((y_trn.shape[0]))

    scaler = MinMaxScaler()
    scaler.fit(X_trn)

    X_trn = scaler.transform(X_trn)

    X_oos = np.array(oos_df[FEA_LIST])
    X_oos = scaler.transform(X_oos)
    y_oos = np.array(oos_df[RSP_VAR])
    y_oos = y_oos.reshape((y_oos.shape[0]))

    clf = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        random_state=1,
        alpha=1.0e-4,
        solver="lbfgs",
        activation="relu",
    )
    clf.fit(X_trn, y_trn)

    y_trn_pred = clf.predict(X_trn)
    y_oos_pred = clf.predict(X_oos)

    conf_mat = confusion_matrix(y_trn, y_trn_pred)

    if False:
        print("Training confusion matrix:", conf_mat)

    conf_mat = confusion_matrix(y_oos, y_oos_pred)

    if False:
        print("Test confusion matrix:", conf_mat)

    trn_df["nn_trend"] = y_trn_pred
    oos_df["nn_trend"] = y_oos_pred

    trn_df["nn_success"] = trn_df.apply(
        lambda x: x.nn_trend * x[RSP_VAR], 
        axis=1,
    )

    oos_df["nn_success"] = oos_df.apply(
        lambda x: x.nn_trend * x[RSP_VAR], 
        axis=1,
    )

    tmp_df = pd.concat([trn_df, oos_df])
    NN_TREND_HASH[symbol] = dict(zip(tmp_df.Date, tmp_df.nn_trend))

    print("\n")
    print("***************************************************************")
    print("Symbol = %s, Num training data = %d" % (symbol, trn_df.shape[0]))
    print(
        "Training Geo only success rate:",
        trn_df[trn_df.geo_success > 0].shape[0] / trn_df.shape[0]
    )
    print(
        "Training MACD only success rate:",
        oos_df[oos_df.macd_success > 0].shape[0] / oos_df.shape[0]
    )
    print(
        "Training Geo/MACD success rate:",
        trn_df[trn_df.geo_macd_success > 0].shape[0] / trn_df.shape[0]
    )
    print(
        "Training Geo/MACD/PTC success rate:",
        trn_df[trn_df.geo_macd_ptc_success > 0].shape[0] / trn_df.shape[0]
    )
    print(
        "Training NN success rate:",
        trn_df[trn_df.nn_success > 0].shape[0] / trn_df.shape[0]
    )

    print("\n")
    print(
        "Test Geo only success rate:",
        oos_df[oos_df.geo_success > 0].shape[0] / oos_df.shape[0]
    )
    print(
        "Test MACD only success rate:",
        oos_df[oos_df.macd_success > 0].shape[0] / oos_df.shape[0]
    )

    print(
        "Test Geo/MACD success rate:",
        oos_df[oos_df.geo_macd_success > 0].shape[0] / oos_df.shape[0]
    )
    print(
        "Test Geo/MACD/PTC success rate:",
        oos_df[oos_df.geo_macd_ptc_success > 0].shape[0] / oos_df.shape[0]
    )
    print(
        "Test NN success rate:",
        oos_df[oos_df.nn_success > 0].shape[0] / oos_df.shape[0]
    )
    print("***************************************************************")

    return (clf.score(X_trn, y_trn), clf.score(X_oos, y_oos))

# ***********************************************************************
# Main
# ***********************************************************************

if __name__ == "__main__":

    beg_time = time.time()

    trn_scores = []    
    oos_scores = []
    for symbol in SYMBOLS:
        tmp = do_nn_class(symbol, HIDDEN_LAYER_SIZES)
        trn_scores.append(tmp[0])
        oos_scores.append(tmp[1])

    print("\n")
    print("Total runtime: %0.2f seconds!" % (time.time()-beg_time))
        
    print("\n")
    print(
        "Mean training score: %0.2f +/- %0.2f" % \
        (np.mean(trn_scores), np.std(trn_scores))
    )    
    print(
        "Mean test score: %0.2f +/- %0.2f" % \
        (np.mean(oos_scores), np.std(oos_scores))
    )

    if BUILD_PRT_FLAG:
        
        inp_prt_hash = json.load(open(INP_PRT_FILE, "r"))
        out_prt_hash = {}
        
        for snap_date in inp_prt_hash:
            
            # if snap_date <= MAX_TRN_DATE:
            #     continue
            
            out_prt_hash[snap_date] = {}
            
            for symbol in inp_prt_hash[snap_date]:
                
                if symbol in NN_TREND_HASH and snap_date in NN_TREND_HASH[symbol]:
                    out_prt_hash[snap_date][symbol] = np.sign(
                        NN_TREND_HASH[symbol][snap_date]
                    ) * abs(inp_prt_hash[snap_date][symbol])
                else:
                    out_prt_hash[snap_date][symbol] = inp_prt_hash[snap_date][symbol]
                    
        json.dump(out_prt_hash, open(OUT_PRT_FILE, "w"))
