# -*- coding: utf-8 -*-
"""v26-B:卡×商户中间粒度元特征(docs/202608/04-v26_分类器攻坚.md)。

冠军引用的 Home Credit 17th 原帖:行单元 = previous application(每人几条,行特征丰富)。
v8 复现用原始交易行(每卡上千条,行特征薄,行级 AUC 0.85-0.89)——粒度错配才是失败主因。
Elo 的对应中间粒度 = 卡×商户单元(每卡几十个,单元特征 16 列可富集)。
单元级 LGB(卡级折外,负例 25% 下采样)→ 卡级聚合 6 列 → clf 判据(AUC + 融合替换)。
用法:ELO_SEED=777 python src/v26b_unitmeta.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, roc_auc_score

import elo_pipeline as ep
import v11_formula as v11
import v5_fusion as vf

UNIT_CACHE = "outputs/unit_cm.parquet"
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def build_units() -> pd.DataFrame:
    if os.path.exists(UNIT_CACHE):
        return pd.read_parquet(UNIT_CACHE)
    cols = ["card_id", "merchant_id", "purchase_amount", "purchase_date", "month_lag",
            "installments", "category_1"]
    hist = ep.clean_transactions(ep.load_transactions("historical_transactions.csv"))
    hist = hist[hist["authorized_flag"] == 1][cols]
    new = ep.clean_transactions(ep.load_transactions("new_merchant_transactions.csv"))[cols]
    tx = pd.concat([hist, new], ignore_index=True)
    del hist, new
    tx["card_id"] = tx["card_id"].astype(str)
    tx["merchant_id"] = tx["merchant_id"].astype(str)
    amt = np.round(tx["purchase_amount"].to_numpy(np.float64) / 0.00150265118 + 497.06, 2)
    tx["amt"] = amt
    tx["is_int"] = (np.round(amt * 100).astype(np.int64) % 100 == 0)
    ref = pd.Timestamp(ep.CONFIG["REF_DATE"])
    tx["days"] = (ref - tx["purchase_date"]).dt.total_seconds() / 86400.0
    tx["is_new"] = (tx["month_lag"] > 0)

    mer_pop = tx.groupby("merchant_id")["card_id"].nunique().rename("u_mer_ncards")
    g = tx.groupby(["card_id", "merchant_id"], sort=False)
    u = g.agg(u_n=("amt", "size"), u_amt_sum=("amt", "sum"), u_amt_mean=("amt", "mean"),
              u_amt_max=("amt", "max"), u_ml_min=("month_lag", "min"), u_ml_max=("month_lag", "max"),
              u_nmonth=("month_lag", "nunique"), u_days_min=("days", "min"), u_days_max=("days", "max"),
              u_inst_mean=("installments", "mean"), u_cat1_mean=("category_1", "mean"),
              u_int_share=("is_int", "mean"), u_n_new=("is_new", "sum")).reset_index()
    u["u_span"] = u["u_ml_max"] - u["u_ml_min"]
    u["u_density"] = u["u_n"] / (u["u_span"] + 1)
    u = u.merge(mer_pop, on="merchant_id", how="left")
    u.drop(columns="merchant_id", inplace=True)
    u.to_parquet(UNIT_CACHE)
    log(f"单元表缓存 {UNIT_CACHE}: {u.shape}")
    return u


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    u = build_units()
    ucols = [c for c in u.columns if c != "card_id"]
    base = pd.read_parquet("data/processed/features.parquet")
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    folds = ep.make_folds(y)
    ybin = (y < -30).astype(int)
    log(f"单元 {u.shape},train 卡 {len(train)}")

    # 卡→折号;test 卡 = -1
    fold_of = pd.Series(-1, index=test["card_id"].astype(str))
    fmap = pd.Series(index=train["card_id"].astype(str), dtype=int)
    for k, (_, va) in enumerate(folds):
        fmap.iloc[va] = k
    lab = pd.Series(ybin.to_numpy(), index=train["card_id"].astype(str))
    u["fold"] = u["card_id"].map(pd.concat([fmap, fold_of])).fillna(-9).astype(int)
    u = u[u["fold"] != -9].reset_index(drop=True)          # 只保留 train/test 卡
    u["lab"] = u["card_id"].map(lab)                        # test 单元为 NaN
    rng = np.random.default_rng(777)

    params = {**ep.CLF_PARAMS} if hasattr(ep, "CLF_PARAMS") else {**ep.LGB_PARAMS,
              "objective": "binary", "metric": "auc"}
    u["um_pred"] = 0.0
    te_mask = u["fold"] == -1
    Xte_u = u.loc[te_mask, ucols]
    for k in range(len(folds)):
        trm = (u["fold"] != k) & (~te_mask)
        # 负例 25% 下采样提速(正例全保留)
        neg = trm & (u["lab"] == 0)
        keep = rng.random(len(u)) < 0.25
        trm2 = (trm & (u["lab"] == 1)) | (neg & keep)
        vam = u["fold"] == k
        m = lgb.train(params, lgb.Dataset(u.loc[trm2, ucols], u.loc[trm2, "lab"]),
                      ep._rounds(3000), valid_sets=[lgb.Dataset(u.loc[vam, ucols], u.loc[vam, "lab"])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        u.loc[vam, "um_pred"] = m.predict(u.loc[vam, ucols], num_iteration=m.best_iteration)
        u.loc[te_mask, "um_pred"] += m.predict(Xte_u, num_iteration=m.best_iteration) / len(folds)
        a = roc_auc_score(u.loc[vam, "lab"], u.loc[vam, "um_pred"])
        log(f"  [unit] fold{k + 1}: 单元级 AUC={a:.5f} iter={m.best_iteration} 训练行 {int(trm2.sum()):,}")

    gb = u.groupby("card_id")["um_pred"]
    agg = pd.DataFrame({"um_mean": gb.mean(), "um_max": gb.max(), "um_std": gb.std(),
                        "um_q90": gb.quantile(0.9)})
    w = u["um_pred"] * u["u_n"]
    agg["um_wmean"] = w.groupby(u["card_id"]).sum() / u.groupby("card_id")["u_n"].sum()
    top3 = u.sort_values("um_pred", ascending=False).groupby("card_id")["um_pred"].head(3)
    agg["um_top3"] = top3.groupby(u.loc[top3.index, "card_id"]).mean()
    agg = agg.reset_index()
    tr_m = train[["card_id"]].astype(str).merge(agg, on="card_id", how="left")
    log("um 分布 train/test 均值对照: " +
        ", ".join(f"{c} {tr_m[c].mean():.4f}/{test[['card_id']].astype(str).merge(agg, on='card_id', how='left')[c].mean():.4f}"
                  for c in ("um_mean", "um_max")))

    # 卡级 clf:fm X + um 6 列
    base2 = base.merge(v11.hist_lag_amt(), on="card_id", how="left")
    base2["card_id"] = base2["card_id"].astype(str)
    base2 = base2.merge(agg, on="card_id", how="left")
    train2 = base2[base2["is_train"] == 1].reset_index(drop=True)
    test2 = base2[base2["is_train"] == 0].reset_index(drop=True)
    fm_tr, fm_te = v11.formula_block(train2), v11.formula_block(test2)
    imp = pd.read_csv("outputs/feature_importance.csv")
    sel = [c for c in imp[imp["gain"] > 0].head(ep.CONFIG["TOP_K"])["feature"] if c in train2.columns]
    z = np.load("outputs/te_features.npz", allow_pickle=True)
    te_names = [str(x) for x in z["names"]]
    td = pd.read_parquet("outputs/td_features.parquet")
    um_cols = [c for c in agg.columns if c != "card_id"]

    def asm(side, zte, fm):
        m1 = side[["card_id"]].merge(td, on="card_id", how="left").drop(columns="card_id")
        return pd.concat([side[sel].reset_index(drop=True), pd.DataFrame(zte, columns=te_names),
                          m1.astype(np.float32), fm.reset_index(drop=True),
                          side[um_cols].reset_index(drop=True).astype(np.float32)], axis=1)

    X, X_test = asm(train2, z["tr"], fm_tr), asm(test2, z["te"], fm_te)
    log(f"卡级 X={X.shape}(含 um {len(um_cols)});训练 clf")
    oof, pred, auc = ep.cv_outlier_clf(X, y, X_test, folds)
    os.makedirs("outputs/base_um", exist_ok=True)
    np.savez("outputs/base_um/clf_um.npz", oof=oof, pred=pred)
    log(f"clf+um AUC={auc:.5f} vs f_clf 0.90586(冠军对标 0.914)")

    bases = vf.load_bases()
    yb = ybin.to_numpy()
    REG = [k for k in ("lgb", "xgb", "cat", "hub", "mlp", "mlp2", "et") if k in bases]
    T = [k for k in ("t_lgb", "t_xgb", "t_cat", "t_hub") if k in bases]
    D = [k for k in ("d_lgb", "d_xgb", "d_cat", "d_hub") if k in bases]
    F = [k for k in ("f_lgb", "f_xgb", "f_cat", "f_hub") if k in bases]
    N = sorted(k for k in bases if k.startswith("n_"))
    allf = (REG + T + D + F + ["t_clf", "t_clean", "d_clf", "d_clean",
            "f_clf", "f_clean", "p_cal", "ev", "p_cal_x_clean"] + N)
    r0, _, _ = vf.evaluate(allf, "bayes", bases, y, yb, folds, p_src="f_clf", clean_src="f_clean")
    bases["f_clf"] = (oof, pred)
    r1, _, _ = vf.evaluate(allf, "bayes", bases, y, yb, folds, p_src="f_clf", clean_src="f_clean")
    d = r0 - r1
    log(f"判据[v26-B unit-meta 替换]:基线={r0:.5f} 替换后={r1:.5f} → Δ={d:+.5f} "
        f"{'✅ 通过' if d > 0.0005 else '❌ 不足'}")
    if d <= 0.0005:
        sys.exit(3)


if __name__ == "__main__":
    main()
