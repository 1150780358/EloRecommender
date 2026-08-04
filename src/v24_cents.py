# -*- coding: utf-8 -*-
"""v24-A:金额数字结构特征族(自有创新;docs/202608/04-v24_自有创新.md)。

raddar 解码(ep 已内置量级恢复)之上,首次利用**数字结构语义**:
整数金额/整十整百 = 充值·账单;.99/.90 = 零售定价;同一精确金额反复 = 订阅/固定账单。
hist/new 双窗 × ~14 列:int/round5/round10/round50/c99/c90 占比、cents nunique 与熵、
金额 nunique 比、众数金额占比与整额性、众数 cents。
判据:单模 lgb(sel+TE+td+fm+ct)vs outputs/base_fm/lgb.npz(3.63246)>0.0005。
用法:ELO_SEED=777 python src/v24_cents.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import elo_pipeline as ep
import v11_formula as v11

CT_CACHE = "outputs/cents_features.parquet"
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def digit_block(tx: pd.DataFrame, pfx: str) -> pd.DataFrame:
    """解码金额的数字结构按卡聚合。"""
    amt = np.round(tx["purchase_amount"].to_numpy(np.float64) / 0.00150265118 + 497.06, 2)
    cents = np.round(amt * 100).astype(np.int64) % 100
    intp = np.round(amt).astype(np.int64)
    g = pd.DataFrame({
        "card_id": tx["card_id"].to_numpy(),
        "amt2": np.round(amt, 2),
        "is_int": (cents == 0),
        "r5": (cents == 0) & (intp % 5 == 0),
        "r10": (cents == 0) & (intp % 10 == 0),
        "r50": (cents == 0) & (intp % 50 == 0),
        "c99": (cents == 99),
        "c90": (cents == 90),
        "cents": cents,
    })
    gb = g.groupby("card_id", sort=False)
    out = gb.agg(**{
        f"{pfx}_int_share": ("is_int", "mean"),
        f"{pfx}_r5_share": ("r5", "mean"),
        f"{pfx}_r10_share": ("r10", "mean"),
        f"{pfx}_r50_share": ("r50", "mean"),
        f"{pfx}_c99_share": ("c99", "mean"),
        f"{pfx}_c90_share": ("c90", "mean"),
        f"{pfx}_cents_nuniq": ("cents", "nunique"),
        f"{pfx}_cents_mode": ("cents", lambda s: s.mode().iat[0]),
        f"{pfx}_n": ("cents", "size"),
    })
    # cents 熵
    ce = g.groupby(["card_id", "cents"], sort=False).size().rename("k").reset_index()
    ce = ce.merge(out[f"{pfx}_n"].rename("n"), on="card_id")
    p = ce["k"] / ce["n"]
    ent = (-p * np.log(p)).groupby(ce["card_id"]).sum().rename(f"{pfx}_cents_entropy")
    # 精确同额重复:众数金额占比、金额 nunique 比、众数金额是否整额
    am = g.groupby(["card_id", "amt2"], sort=False).size().rename("k").reset_index()
    top = am.sort_values("k", ascending=False).drop_duplicates("card_id").set_index("card_id")
    rep = pd.DataFrame({
        f"{pfx}_amt_top_share": top["k"] / out[f"{pfx}_n"],
        f"{pfx}_amt_top_isint": (np.round(top["amt2"] * 100).astype(np.int64) % 100 == 0).astype(np.int8),
        f"{pfx}_amt_nuniq_ratio": am.groupby("card_id")["k"].size() / out[f"{pfx}_n"],
    })
    res = out.drop(columns=f"{pfx}_n").join(ent).join(rep).reset_index()
    res["card_id"] = res["card_id"].astype(str)
    return res


def build_ct() -> pd.DataFrame:
    if os.path.exists(CT_CACHE):
        return pd.read_parquet(CT_CACHE)
    cols = ["card_id", "purchase_amount", "authorized_flag"]
    hist = ep.clean_transactions(ep.load_transactions("historical_transactions.csv"))
    hist = hist[hist["authorized_flag"] == 1][cols]
    new = ep.clean_transactions(ep.load_transactions("new_merchant_transactions.csv"))[cols]
    ct = digit_block(hist, "cth").merge(digit_block(new, "ctn"), on="card_id", how="outer")
    ct.to_parquet(CT_CACHE)
    log(f"数字结构特征缓存 {CT_CACHE}: {ct.shape}")
    return ct


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    ct = build_ct()
    base = pd.read_parquet("data/processed/features.parquet")
    base = base.merge(v11.hist_lag_amt(), on="card_id", how="left")
    base = base.merge(ct, on="card_id", how="left")
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
    ct_cols = [c for c in ct.columns if c != "card_id"]

    def asm(side, zte, fm):
        m1 = side[["card_id"]].merge(td, on="card_id", how="left").drop(columns="card_id")
        return pd.concat([side[sel].reset_index(drop=True), pd.DataFrame(zte, columns=te_names),
                          m1.astype(np.float32), fm.reset_index(drop=True),
                          side[ct_cols].reset_index(drop=True).astype(np.float32)], axis=1)

    X, X_test = asm(train, z["tr"], fm_tr), asm(test, z["te"], fm_te)
    log(f"X={X.shape}(含 ct {len(ct_cols)})")
    oof, pred, _, gain = ep.cv_lightgbm(X, y, X_test, folds, ep.LGB_PARAMS, "lgb+ct")
    os.makedirs("outputs/base_ct", exist_ok=True)
    np.savez("outputs/base_ct/lgb.npz", oof=oof, pred=pred)
    ref = rmse(y, np.load("outputs/base_fm/lgb.npz")["oof"])
    s = rmse(y, oof)
    d = ref - s
    g2 = pd.DataFrame({"feature": X.columns, "gain": gain}).sort_values("gain", ascending=False)
    in50 = int(g2.head(50)["feature"].str.startswith(("cth_", "ctn_")).sum())
    log(f"ct 列进 gain 前 50:{in50};前 5:{list(g2.head(5)['feature'])}")
    log(f"判据[v24-A cents]:OOF={s:.5f} vs fm 基线 {ref:.5f} → 改善 {d:+.5f} "
        f"{'✅ 通过' if d > 0.0005 else '❌ 不足'}")
    if d <= 0.0005:
        sys.exit(3)


if __name__ == "__main__":
    main()
