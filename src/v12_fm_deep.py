# -*- coding: utf-8 -*-
"""v12:fm 族纵深 —— retention 结构化代理(docs/202608/03-v12_fm纵深.md)。

官方口径 x = future spending + retention;v11 的 16 列覆盖 spending(总量比值),
本脚本补 retention 组件的三族代理(全部 log2(r+1e-10) 哨兵同构):
  A. 交叉 retention:new 窗消费落在历史 top3 品类 / top1 城市的金额比值与占比;
  B. 重访结构:new 窗内访问≥2次商户的金额/商户数、lag1→lag2 商户留存、活跃周数;
  C. 合成分子:x̂ = (new窗金额/2 + 历史回头商户近3月月均) ÷ 基线 —— 用历史回头率
     外推 new 表缺失的"评估窗回头交易",逼近真分子。

判据:lgb(sel+TE+td+fm+fd)OOF vs outputs/base_fm/lgb.npz(777,3.63246)改善>0.0005。
用法:ELO_SEED=777 python src/v12_fm_deep.py [lgb|rest|all]
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

import elo_pipeline as ep
import v11_formula as v11

OUT_DIR = "outputs/base_fd"
REF_LGB = "outputs/base_fm/lgb.npz"
FD_CACHE = "outputs/fm_deep.parquet"
EPS = 1e-10
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def deep_cache() -> pd.DataFrame:
    """hist/new 原始表 → card 级 retention 原料(全为分子类,分母走 features 现有列)。"""
    if os.path.exists(FD_CACHE):
        return pd.read_parquet(FD_CACHE)
    hist = ep.clean_transactions(ep.load_transactions("historical_transactions.csv"))
    hist = hist[hist["authorized_flag"] == 1]
    log(f"hist auth=1: {hist.shape}")

    # 历史 top3 品类 / top1 城市(按金额)
    hc = hist.groupby(["card_id", "merchant_category_id"], observed=True)["purchase_amount"].sum().reset_index()
    hc["rk"] = hc.groupby("card_id")["purchase_amount"].rank(ascending=False, method="first")
    top_cat = hc[hc["rk"] <= 3][["card_id", "merchant_category_id"]]
    hy = hist.groupby(["card_id", "city_id"], observed=True)["purchase_amount"].sum().reset_index()
    hy["rk"] = hy.groupby("card_id")["purchase_amount"].rank(ascending=False, method="first")
    top_city = hy[hy["rk"] <= 1][["card_id", "city_id"]]
    log(f"top3 品类对 {top_cat.shape},top1 城市对 {top_city.shape}")

    # 历史重访商户(该卡访问≥2次):金额占比、全期月均、近3月月均
    hm = hist.groupby(["card_id", "merchant_id"], observed=True).agg(
        amt=("purchase_amount", "sum"), n=("purchase_amount", "size")).reset_index()
    rep_pairs = hm[hm["n"] >= 2][["card_id", "merchant_id"]]
    months = hist.groupby("card_id", observed=True)["month_lag"].agg(["min", "max"])
    months = (months["max"] - months["min"] + 1).clip(lower=1).rename("hd_months")
    tot = hist.groupby("card_id", observed=True)["purchase_amount"].sum().rename("hd_amt")
    rep_amt = hm[hm["n"] >= 2].groupby("card_id", observed=True)["amt"].sum().rename("hd_rep_amt")
    h3 = hist[hist["month_lag"] >= -2].merge(rep_pairs, on=["card_id", "merchant_id"], how="inner")
    rep_r3 = h3.groupby("card_id", observed=True)["purchase_amount"].sum().rename("hd_rep_r3")
    log(f"重访商户对 {rep_pairs.shape}")
    del hc, hy, hm, h3, hist

    new = ep.clean_transactions(ep.load_transactions("new_merchant_transactions.csv"))
    log(f"new: {new.shape}")
    nd_amt = new.groupby("card_id", observed=True)["purchase_amount"].sum().rename("nd_amt")
    # A. 交叉 retention:new 窗落在历史熟悉维度的金额
    ncat = new.merge(top_cat, on=["card_id", "merchant_category_id"], how="inner")
    nd_retcat = ncat.groupby("card_id", observed=True)["purchase_amount"].sum().rename("nd_retcat_amt")
    ncity = new.merge(top_city, on=["card_id", "city_id"], how="inner")
    nd_retcity = ncity.groupby("card_id", observed=True)["purchase_amount"].sum().rename("nd_retcity_amt")
    # B. 重访结构:new 窗内访问≥2次的商户;lag1 与 lag2 都出现的商户
    nm = new.groupby(["card_id", "merchant_id"], observed=True).agg(
        amt=("purchase_amount", "sum"), n=("purchase_amount", "size"),
        l_min=("month_lag", "min"), l_max=("month_lag", "max")).reset_index()
    rev = nm[nm["n"] >= 2].groupby("card_id", observed=True)
    nd_rev_amt = rev["amt"].sum().rename("nd_rev_amt")
    nd_rev_mer = rev["amt"].size().astype(np.float64).rename("nd_rev_mer")
    nd_l12 = (nm[(nm["l_min"] == 1) & (nm["l_max"] == 2)].groupby("card_id", observed=True)["amt"]
              .size().astype(np.float64).rename("nd_l12_mer"))
    # 活跃周数(ISO 年×周去重)
    wk = new["purchase_date"].dt.isocalendar()
    new["yw"] = wk["year"].astype(np.int32) * 100 + wk["week"].astype(np.int32)
    nd_wk = new.groupby("card_id", observed=True)["yw"].nunique().astype(np.float64).rename("nd_wk")

    parts = [months, tot, rep_amt, rep_r3, nd_amt, nd_retcat, nd_retcity,
             nd_rev_amt, nd_rev_mer, nd_l12, nd_wk]
    res = pd.concat([p.set_axis(p.index.astype(str)) for p in parts], axis=1).fillna(0.0)
    res.index.name = "card_id"
    res = res.reset_index()
    res.to_parquet(FD_CACHE)
    log(f"retention 原料缓存 {FD_CACHE}: {res.shape}")
    return res


def deep_block(df: pd.DataFrame) -> pd.DataFrame:
    """12 列 retention 代理,log2(r+1e-10),无 new 活动 → -33.2 与哨兵同构。"""
    g = lambda c: df[c].fillna(0).to_numpy(np.float64) if c in df else np.zeros(len(df))
    lg = lambda r: np.log2(np.clip(r, 0, None) + EPS).astype(np.float32)
    div = lambda a, b: np.where(b > 1e-9, a / np.where(b > 1e-9, b, 1), 0.0)
    months = np.clip(g("hd_months"), 1, None)
    h_amt_m = np.clip(g("hd_amt") / months, 1e-9, None)          # 历史月均消费(auth=1)
    rep_m = g("hd_rep_amt") / months                              # 历史回头月均
    rep_r3_m = g("hd_rep_r3") / 3                                 # 近3月回头月均
    r3a = np.clip(g("hl_r3_amt"), 1e-9, None)                     # v11 近3月总基线
    n_amt = g("nd_amt")

    f = pd.DataFrame(index=df.index)
    # A. 交叉 retention
    f["fd_ret_cat"] = lg(div(g("nd_retcat_amt") / 2, h_amt_m))
    f["fd_ret_city"] = lg(div(g("nd_retcity_amt") / 2, h_amt_m))
    f["fd_ret_cat_share"] = lg(div(g("nd_retcat_amt"), np.clip(n_amt, 1e-9, None)))
    # B. 重访结构
    f["fd_rev"] = lg(div(g("nd_rev_amt") / 2, h_amt_m))
    f["fd_rev_share"] = lg(div(g("nd_rev_amt"), np.clip(n_amt, 1e-9, None)))
    h_mer_m = np.clip(g("hist_merchant_id_nunique") / months, 1e-9, None)
    f["fd_rev_mer"] = lg(div(g("nd_rev_mer") / 2, h_mer_m))
    f["fd_l12"] = lg(div(g("nd_l12_mer"), np.clip(g("newlag1_merchant_id_nunique"), 1, None)))
    f["fd_wk"] = lg(g("nd_wk") / 9)
    # C. 合成分子:补上评估窗回头消费的外推
    f["fd_xhat"] = lg(div(n_amt / 2 + rep_r3_m, h_amt_m))
    f["fd_xhat_r3"] = lg(div(n_amt / 2 + rep_r3_m, r3a))
    f["fd_rep_share"] = lg(div(g("hd_rep_amt"), np.clip(g("hd_amt"), 1e-9, None)))
    f["fd_rep_trend"] = lg(div(rep_r3_m, np.clip(rep_m, 1e-9, None)))
    return f


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    mode = sys.argv[1] if len(sys.argv) > 1 else "lgb"
    base = pd.read_parquet("data/processed/features.parquet")
    base = base.merge(v11.hist_lag_amt(), on="card_id", how="left")
    base = base.merge(deep_cache(), on="card_id", how="left")
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    folds = ep.make_folds(y)
    fm_tr, fm_te = v11.formula_block(train), v11.formula_block(test)
    fd_tr, fd_te = deep_block(train), deep_block(test)
    ok = (y > -30).to_numpy()
    log("fd 特征与 target 的 spearman(全量 | 仅非 outlier):")
    for c in fd_tr.columns:
        log(f"  {c:16s} {spearmanr(fd_tr[c], y).statistic:+.4f} | "
            f"{spearmanr(fd_tr[c][ok], y[ok]).statistic:+.4f}")
    imp = pd.read_csv("outputs/feature_importance.csv")
    sel = [c for c in imp[imp["gain"] > 0].head(ep.CONFIG["TOP_K"])["feature"] if c in train.columns]
    z = np.load("outputs/te_features.npz", allow_pickle=True)
    te_names = [str(x) for x in z["names"]]
    td = pd.read_parquet("outputs/td_features.parquet")

    def assemble(side, zte, fm, fd):
        m1 = side[["card_id"]].merge(td, on="card_id", how="left").drop(columns="card_id")
        return pd.concat([side[sel].reset_index(drop=True), pd.DataFrame(zte, columns=te_names),
                          m1.astype(np.float32), fm.reset_index(drop=True), fd.reset_index(drop=True)], axis=1)

    X = assemble(train, z["tr"], fm_tr, fd_tr)
    X_test = assemble(test, z["te"], fm_te, fd_te)
    log(f"X={X.shape}(sel {len(sel)} + TE {len(te_names)} + td {td.shape[1] - 1}"
        f" + fm {fm_tr.shape[1]} + fd {fd_tr.shape[1]})")
    os.makedirs(OUT_DIR, exist_ok=True)

    def dump(name, oof, pred):
        np.savez(os.path.join(OUT_DIR, f"{name}.npz"), oof=oof, pred=pred)
        log(f"[fd] {name:6s} OOF={rmse(y, oof):.5f} -> {OUT_DIR}/{name}.npz")

    if mode in ("lgb", "all"):
        oof, pred, _, gain = ep.cv_lightgbm(X, y, X_test, folds, ep.LGB_PARAMS, "lgb+fd")
        dump("lgb", oof, pred)
        ref = rmse(y, np.load(REF_LGB)["oof"])
        s = rmse(y, oof)
        d = ref - s
        alarm = "(⚠️ 超警报线,先审计)" if d > 0.003 else ""
        log(f"判据[v12 retention 纵深]:OOF={s:.5f} vs fm 基线 {ref:.5f} → 改善 {d:+.5f} "
            f"{'✅ 通过' if d > 0.0005 else '❌ 不足'}{alarm}")
        g2 = pd.DataFrame({"feature": X.columns, "gain": gain}).sort_values("gain", ascending=False)
        log("fd 列 gain:\n" + g2[g2["feature"].str.startswith("fd_")].to_string(index=False))
        log(f"fd 列进入 gain 前 50 的个数:{int(g2.head(50)['feature'].str.startswith('fd_').sum())}")
        if d <= 0.0005:
            sys.exit(3)   # 判据不足:非零退出,让链条短路

    if mode in ("rest", "all"):
        oof, pred, _ = ep.cv_xgboost(X, y, X_test, folds);            dump("xgb", oof, pred)
        oof, pred, _ = ep.cv_catboost(X, y, X_test, folds);           dump("cat", oof, pred)
        oof, pred, _, _ = ep.cv_lightgbm(X, y, X_test, folds, ep.HUB_PARAMS, "hub+fd"); dump("hub", oof, pred)
        oof, pred, auc = ep.cv_outlier_clf(X, y, X_test, folds)
        np.savez(os.path.join(OUT_DIR, "clf.npz"), oof=oof, pred=pred)
        log(f"[fd] clf AUC={auc:.5f}(fm 版 0.90586)")
        mask = (y < -30).to_numpy()
        oc, pc = np.zeros(len(X)), np.zeros(len(X_test))
        import lightgbm as lgb_
        for k, (tr, va) in enumerate(folds):
            tr_c, va_c = tr[~mask[tr]], va[~mask[va]]
            m = lgb_.train(ep.LGB_PARAMS, lgb_.Dataset(X.iloc[tr_c], y.iloc[tr_c]), 10000,
                           valid_sets=[lgb_.Dataset(X.iloc[va_c], y.iloc[va_c])],
                           callbacks=[lgb_.early_stopping(200, verbose=False)])
            oc[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
            pc += m.predict(X_test, num_iteration=m.best_iteration) / len(folds)
            log(f"  [clean+fd] fold{k + 1} iter={m.best_iteration}")
        dump("clean", oc, pc)


if __name__ == "__main__":
    main()
