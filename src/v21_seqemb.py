# -*- coding: utf-8 -*-
"""v21-B:访问序列 embedding(PPMI-SVD,SGNS 谱等价;docs/202608/03-v21_榜单二次考古.md)。

11th "visit sequence embedding" + 7th w2v 的零依赖实现:
同卡交易序列(去连续重复)窗口 ±2 的 merchant / merchant_category 共现
→ PPMI → TruncatedSVD(mer 16 维 / mcat 8 维)→ 金额加权卡向量 24 列。
判据:单模 lgb(sel+TE+td+fm+emb)vs outputs/base_fm/lgb.npz(3.63246)>0.0005。
用法:ELO_SEED=777 python src/v21_seqemb.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

import elo_pipeline as ep
import v11_formula as v11

EMB_CACHE = "outputs/seq_emb.parquet"
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def ppmi_svd(codes, cards_codes, V, dim, amounts):
    """codes:全交易的 token 码(-1=无效);同卡窗口 ±2 共现 → PPMI → SVD → 卡向量。"""
    ok = codes >= 0
    pairs = []
    for k in (1, 2):
        a, b = codes[:-k], codes[k:]
        same = (cards_codes[:-k] == cards_codes[k:]) & ok[:-k] & ok[k:]
        pairs.append((a[same].astype(np.int64) * V + b[same]).astype(np.int64))
        pairs.append((b[same].astype(np.int64) * V + a[same]).astype(np.int64))
    allp = np.concatenate(pairs)
    log(f"  共现对 {len(allp):,}")
    uniq, cnt = np.unique(allp, return_counts=True)
    rows, cols = (uniq // V).astype(np.int32), (uniq % V).astype(np.int32)
    C = sp.csr_matrix((cnt.astype(np.float64), (rows, cols)), shape=(V, V))
    N = C.sum()
    rs = np.asarray(C.sum(1)).ravel()
    cs = np.asarray(C.sum(0)).ravel()
    P = C.tocoo()
    val = np.log(np.clip(P.data * N / (rs[P.row] * cs[P.col]), 1e-12, None))
    val = np.clip(val, 0, None)                       # PPMI
    M = sp.csr_matrix((val, (P.row, P.col)), shape=(V, V))
    svd = TruncatedSVD(n_components=dim, random_state=777, n_iter=7)
    W = svd.fit_transform(M)                          # [V, dim] token 向量
    log(f"  SVD 完成 explained={svd.explained_variance_ratio_.sum():.3f}")
    # 卡向量:金额加权均值
    w = np.clip(amounts, 0.01, None) * ok
    num = np.zeros((cards_codes.max() + 1, dim))
    den = np.zeros(cards_codes.max() + 1)
    np.add.at(num, cards_codes[ok], W[codes[ok]] * w[ok, None])
    np.add.at(den, cards_codes[ok], w[ok])
    return num / np.clip(den[:, None], 1e-9, None)


def build_emb() -> pd.DataFrame:
    if os.path.exists(EMB_CACHE):
        return pd.read_parquet(EMB_CACHE)
    cols = ["card_id", "purchase_date", "purchase_amount", "merchant_id", "merchant_category_id"]
    hist = ep.clean_transactions(ep.load_transactions("historical_transactions.csv"))
    hist = hist[hist["authorized_flag"] == 1][cols]
    new = ep.clean_transactions(ep.load_transactions("new_merchant_transactions.csv"))[cols]
    tx = pd.concat([hist, new], ignore_index=True)
    del hist, new
    tx["card_id"] = tx["card_id"].astype(str)
    tx = tx.sort_values(["card_id", "purchase_date"], kind="mergesort").reset_index(drop=True)
    cards_codes, card_vals = pd.factorize(tx["card_id"])
    cards_codes = cards_codes.astype(np.int32)
    log(f"交易 {tx.shape} 卡数 {len(card_vals)}")

    out = {}
    for name, col, dim, min_cnt in [("mer", "merchant_id", 16, 5),
                                    ("mcat", "merchant_category_id", 8, 1)]:
        codes, vals = pd.factorize(tx[col])
        vc = np.bincount(codes[codes >= 0], minlength=len(vals))
        keep = vc >= min_cnt
        remap = np.full(len(vals), -1, np.int32)
        remap[keep] = np.arange(int(keep.sum()), dtype=np.int32)
        codes2 = np.where(codes >= 0, remap[np.clip(codes, 0, None)], -1).astype(np.int32)
        # 去连续重复(同卡同 token 连续出现只保留首个)
        dup = np.zeros(len(tx), bool)
        dup[1:] = (codes2[1:] == codes2[:-1]) & (cards_codes[1:] == cards_codes[:-1])
        c3 = np.where(dup, -1, codes2)
        V = int(keep.sum())
        log(f"[{name}] vocab={V}(min_count={min_cnt})")
        cv = ppmi_svd(c3, cards_codes, V, dim, tx["purchase_amount"].to_numpy(np.float64))
        for i in range(dim):
            out[f"se_{name}_{i}"] = cv[:, i].astype(np.float32)
    res = pd.DataFrame(out)
    res.insert(0, "card_id", card_vals.astype(str))
    res.to_parquet(EMB_CACHE)
    log(f"embedding 缓存 {EMB_CACHE}: {res.shape}")
    return res


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    emb = build_emb()
    base = pd.read_parquet("data/processed/features.parquet")
    base = base.merge(v11.hist_lag_amt(), on="card_id", how="left")
    base = base.merge(emb, on="card_id", how="left")
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
    se_cols = [c for c in emb.columns if c != "card_id"]

    def asm(side, zte, fm):
        m1 = side[["card_id"]].merge(td, on="card_id", how="left").drop(columns="card_id")
        return pd.concat([side[sel].reset_index(drop=True), pd.DataFrame(zte, columns=te_names),
                          m1.astype(np.float32), fm.reset_index(drop=True),
                          side[se_cols].reset_index(drop=True)], axis=1)

    X, X_test = asm(train, z["tr"], fm_tr), asm(test, z["te"], fm_te)
    log(f"X={X.shape}(含 se {len(se_cols)})")
    oof, pred, _, gain = ep.cv_lightgbm(X, y, X_test, folds, ep.LGB_PARAMS, "lgb+se")
    os.makedirs("outputs/base_se", exist_ok=True)
    np.savez("outputs/base_se/lgb.npz", oof=oof, pred=pred)
    ref = rmse(y, np.load("outputs/base_fm/lgb.npz")["oof"])
    s = rmse(y, oof)
    d = ref - s
    log(f"判据[v21-B seq-emb]:OOF={s:.5f} vs fm 基线 {ref:.5f} → 改善 {d:+.5f} "
        f"{'✅ 通过' if d > 0.0005 else '❌ 不足'}")
    g2 = pd.DataFrame({"feature": X.columns, "gain": gain}).sort_values("gain", ascending=False)
    log(f"se 列进入 gain 前 50:{int(g2.head(50)['feature'].str.startswith('se_').sum())}")
    if d <= 0.0005:
        sys.exit(3)


if __name__ == "__main__":
    main()
