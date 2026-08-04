# -*- coding: utf-8 -*-
"""v22-A:18th 时间预测残差特征(docs/202608/04-v22_低先验备选出清.md)。

18th ⑤ 配方:辅助模型用 hist-only 特征预测 new 窗末次交易时点,残差入主模型。
aux 目标 = REF_DATE − 卡末次 new 交易日(天);aux 特征排除一切含 new 信息的列;
train+test 合并 5 折 OOF 化(目标不含 target,无泄露),残差对全卡可得。
判据:单模 lgb(sel+TE+td+fm+tp)vs outputs/base_fm/lgb.npz(3.63246)>0.0005。
用法:ELO_SEED=777 python src/v22_timeres.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import elo_pipeline as ep
import v11_formula as v11

rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def build_timeres(base: pd.DataFrame) -> pd.DataFrame:
    """aux 模型 5 折 OOF 预测末次 new 交易时点,返回 card_id + tp_last/tp_pred/tp_resid。"""
    new = ep.clean_transactions(ep.load_transactions("new_merchant_transactions.csv"))
    new["card_id"] = new["card_id"].astype(str)
    last = new.groupby("card_id")["purchase_date"].max().rename("last_new")
    ref = pd.Timestamp(ep.CONFIG["REF_DATE"])
    aux_y = ((ref - last).dt.total_seconds() / 86400.0).rename("tp_last")

    df = base.merge(aux_y, on="card_id", how="left")
    drop = {"card_id", "target", "is_train", "tp_last"}
    feats = [c for c in df.columns
             if c not in drop and "new" not in c and not c.startswith("x_")
             and pd.api.types.is_numeric_dtype(df[c])]
    log(f"aux 特征 {len(feats)} 列(hist-only+静态)")

    has = df["tp_last"].notna().to_numpy()
    Xa, ya = df.loc[has, feats], df.loc[has, "tp_last"]
    params = {**ep.LGB_PARAMS, "objective": "regression", "metric": "rmse"}
    pred = np.full(len(df), np.nan)
    idx = np.where(has)[0]
    for k, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=777).split(Xa)):
        m = lgb.train(params, lgb.Dataset(Xa.iloc[tr], ya.iloc[tr]), 10000,
                      valid_sets=[lgb.Dataset(Xa.iloc[va], ya.iloc[va])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        pred[idx[va]] = m.predict(Xa.iloc[va], num_iteration=m.best_iteration)
        log(f"  [aux] fold{k + 1}: rmse={rmse(ya.iloc[va], pred[idx[va]]):.3f}d iter={m.best_iteration}")
    out = pd.DataFrame({"card_id": df["card_id"],
                        "tp_last": df["tp_last"].astype(np.float32),
                        "tp_pred": pred.astype(np.float32)})
    out["tp_resid"] = out["tp_last"] - out["tp_pred"]
    log(f"aux OOF rmse={rmse(ya, pred[idx]):.3f}d 覆盖率={has.mean():.3f}")
    return out


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    base = pd.read_parquet("data/processed/features.parquet")
    base = base.merge(v11.hist_lag_amt(), on="card_id", how="left")
    tp = build_timeres(base)
    base = base.merge(tp, on="card_id", how="left")
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    folds = ep.make_folds(y)
    fm_tr, fm_te = v11.formula_block(train), v11.formula_block(test)
    imp = pd.read_csv("outputs/feature_importance.csv")
    sel = [c for c in imp[imp["gain"] > 0].head(ep.CONFIG["TOP_K"])["feature"] if c in train.columns]
    z = np.load("outputs/te_features.npz", allow_pickle=True)
    te_names = [str(x) for x in z["names"]]
    td = pd.read_parquet("outputs/td_features.parquet")
    tp_cols = ["tp_last", "tp_pred", "tp_resid"]

    def asm(side, zte, fm):
        m1 = side[["card_id"]].merge(td, on="card_id", how="left").drop(columns="card_id")
        return pd.concat([side[sel].reset_index(drop=True), pd.DataFrame(zte, columns=te_names),
                          m1.astype(np.float32), fm.reset_index(drop=True),
                          side[tp_cols].reset_index(drop=True)], axis=1)

    X, X_test = asm(train, z["tr"], fm_tr), asm(test, z["te"], fm_te)
    log(f"X={X.shape}(含 tp 3)")
    oof, pred, _, gain = ep.cv_lightgbm(X, y, X_test, folds, ep.LGB_PARAMS, "lgb+tp")
    os.makedirs("outputs/base_tp", exist_ok=True)
    np.savez("outputs/base_tp/lgb.npz", oof=oof, pred=pred)
    ref = rmse(y, np.load("outputs/base_fm/lgb.npz")["oof"])
    s = rmse(y, oof)
    d = ref - s
    g2 = pd.DataFrame({"feature": X.columns, "gain": gain}).sort_values("gain", ascending=False)
    rk = {c: int((g2["feature"] == c).to_numpy().argmax()) + 1 for c in tp_cols}
    log(f"tp 列 gain 排名:{rk}")
    log(f"判据[v22-A timeres]:OOF={s:.5f} vs fm 基线 {ref:.5f} → 改善 {d:+.5f} "
        f"{'✅ 通过' if d > 0.0005 else '❌ 不足'}")
    if d <= 0.0005:
        sys.exit(3)


if __name__ == "__main__":
    main()
