# -*- coding: utf-8 -*-
"""v15:NN 通道扩展 × outlier 分类头(docs/202608/03-v15_通道扩展与clf头.md)。

A `reg`:13 通道 2 层 GRU × 5 seed → outputs/base_nn/gru_x2.npz(回归成员,NREG 自动吸收)
B `clf`:同架构 BCE 分类头,早停按 AUC × 5 seed → outputs/base_nn_clf/clf.npz(独立目录!)
用法:ELO_SEED=777 python src/v15_nn2.py data
     ELO_SEED=777 python src/v15_nn2.py reg|clf <dev_id>
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score

import elo_pipeline as ep
import v13_gru as v13

PARTS_DIR = "outputs/nn_parts"
OUT_REG = "outputs/base_nn"
OUT_CLF = "outputs/base_nn_clf"
SEQ2_CACHE = "outputs/seq_tensor_ext2.npz"
SEEDS = [777, 1777, 2777, 3777, 4777]
LAGS = v13.LAGS
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def build_seq_ext2():
    """13 通道:v14 ext 10 通道 + 商户新鲜度 + 金额 std + 周末占比。"""
    base = pd.read_parquet("data/processed/features.parquet")
    cols = ["card_id", "month_lag", "purchase_amount", "merchant_id",
            "installments", "category_1", "city_id", "authorized_flag", "purchase_date"]
    hist = ep.clean_transactions(ep.load_transactions("historical_transactions.csv"))[cols]
    new = ep.clean_transactions(ep.load_transactions("new_merchant_transactions.csv"))[cols]
    tx = pd.concat([hist, new], ignore_index=True)
    tx["card_id"] = tx["card_id"].astype(str)
    del hist, new
    tx["wkd"] = (tx["purchase_date"].dt.dayofweek >= 5).astype(np.float32)
    first = (tx.groupby(["card_id", "merchant_id"], observed=True)["month_lag"]
             .transform("min") == tx["month_lag"])
    tx["is_first"] = first.astype(np.float32)
    ok = tx[tx["authorized_flag"] == 1]
    g = ok.groupby(["card_id", "month_lag"], observed=True).agg(
        amt=("purchase_amount", "sum"), amx=("purchase_amount", "max"),
        astd=("purchase_amount", "std"),
        cnt=("purchase_amount", "size"), mer=("merchant_id", "nunique"),
        inst=("installments", "mean"), c1=("category_1", "mean"),
        city=("city_id", "nunique"), wkd=("wkd", "mean"))
    fresh_n = (ok[ok["is_first"] > 0]
               .groupby(["card_id", "month_lag"], observed=True)["merchant_id"]
               .nunique().rename("fmer"))
    den = (tx[tx["authorized_flag"] == 0]
           .groupby(["card_id", "month_lag"], observed=True)["purchase_amount"]
           .size().rename("dcnt"))
    g = g.join(fresh_n, how="outer").join(den, how="outer")
    log(f"逐月聚合 {g.shape}")

    piv = {c: g[c].unstack().reindex(columns=LAGS) for c in g.columns}
    cards = piv["amt"].index
    fz = lambda c: piv[c].fillna(0.0).to_numpy(np.float32)
    cnt, dcnt, mer = fz("cnt"), fz("dcnt"), fz("mer")
    chan = [np.log1p(np.clip(fz(c), 0, None)) for c in ("amt", "amx", "cnt", "mer")]
    chan += [(cnt > 0).astype(np.float32), fz("inst"), fz("c1"),
             np.log1p(dcnt), (dcnt / np.clip(cnt + dcnt, 1, None)).astype(np.float32),
             np.log1p(fz("city")),
             (fz("fmer") / np.clip(mer, 1, None)).astype(np.float32),
             np.log1p(np.clip(fz("astd"), 0, None)),
             fz("wkd")]
    seq = np.stack(chan, axis=2)                       # [N,16,13]
    keep = np.array([1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], np.float32)
    idxmap = pd.Index(cards.astype(str))

    base["n_months"] = (base["hist_month_lag_max"].fillna(0)
                        - base["hist_month_lag_min"].fillna(-12) + 1)
    out = {}
    for part, flag in [("tr", 1), ("te", 0)]:
        side = base[base["is_train"] == flag]
        idx = idxmap.get_indexer(side["card_id"].astype(str))
        xs = np.zeros((len(side), len(LAGS), seq.shape[2]), np.float32)
        hit = idx >= 0
        xs[hit] = seq[idx[hit]]
        out[f"xs_{part}"] = xs
        out[f"st_{part}"] = side[v13.STATIC].fillna(0).to_numpy(np.float32)
        log(f"{part}: xs={xs.shape} 无交易卡 {int((~hit).sum())}")
    m = out["xs_tr"].reshape(-1, seq.shape[2])
    mu, sd = m.mean(0), m.std(0) + 1e-6
    for p in ("tr", "te"):
        out[f"xs_{p}"] = (out[f"xs_{p}"] - mu * keep) / np.where(keep > 0, sd, 1.0)
    smu, ssd = out["st_tr"].mean(0), out["st_tr"].std(0) + 1e-6
    for p in ("tr", "te"):
        out[f"st_{p}"] = (out[f"st_{p}"] - smu) / ssd
    np.savez_compressed(SEQ2_CACHE, **out)
    log(f"ext2 序列缓存 {SEQ2_CACHE}: " + ", ".join(f"{k}={v.shape}" for k, v in out.items()))


def train(task, dev_id):
    import torch
    import torch.nn as nn
    dev = f"cuda:{dev_id}" if torch.cuda.is_available() else "cpu"
    z = np.load(SEQ2_CACHE)
    xs_tr, st_tr, xs_te, st_te = z["xs_tr"], z["st_tr"], z["xs_te"], z["st_te"]
    base = pd.read_parquet("data/processed/features.parquet")
    y = base[base["is_train"] == 1].reset_index(drop=True)["target"]
    folds = ep.make_folds(y)
    yv = y.to_numpy(np.float32)
    ybin = (yv < -30).astype(np.float32)
    C, S = xs_tr.shape[2], st_tr.shape[1]
    log(f"[{task}] train {xs_tr.shape} dev={dev} outlier率={ybin.mean():.4f}")

    class SeqNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.GRU(C, 96, num_layers=2, batch_first=True, dropout=0.1)
            self.stat = nn.Linear(S, 32)
            self.head = nn.Sequential(
                nn.Linear(96 + 32, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, xs, st):
            _, hn = self.enc(xs)
            return self.head(torch.cat([hn[-1], torch.relu(self.stat(st))], 1)).squeeze(1)

    XS, ST = torch.from_numpy(xs_tr).to(dev), torch.from_numpy(st_tr).to(dev)
    Y = torch.from_numpy(yv if task == "reg" else ybin).to(dev)
    XSe, STe = torch.from_numpy(xs_te).to(dev), torch.from_numpy(st_te).to(dev)

    def infer(model, xs, st, bs=8192, sig=False):
        model.eval()
        with torch.no_grad():
            o = torch.cat([model(xs[i:i + bs], st[i:i + bs]) for i in range(0, len(xs), bs)])
            if sig:
                o = torch.sigmoid(o)
        return o.float().cpu().numpy()

    os.makedirs(PARTS_DIR, exist_ok=True)
    out_dir = OUT_REG if task == "reg" else OUT_CLF
    os.makedirs(out_dir, exist_ok=True)
    BS, MAX_EP, PAT = 1024, 40, 5
    accum_oof, accum_pred = [], []
    for sd in SEEDS:
        oof, pred = np.zeros(len(yv)), np.zeros(len(xs_te))
        for k, (tr, va) in enumerate(folds):
            torch.manual_seed(sd + k * 101)
            np.random.seed(sd + k * 101)
            model = SeqNet().to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, factor=0.5, patience=2, mode="min" if task == "reg" else "max")
            lossf = nn.MSELoss() if task == "reg" else nn.BCEWithLogitsLoss()
            tr_t = torch.from_numpy(tr).to(dev)
            best = 1e9 if task == "reg" else -1e9
            wait, best_state = 0, None
            for _ in range(MAX_EP):
                model.train()
                perm = tr_t[torch.randperm(len(tr_t), device=dev)]
                for i in range(0, len(perm), BS):
                    b = perm[i:i + BS]
                    opt.zero_grad()
                    loss = lossf(model(XS[b], ST[b]), Y[b])
                    loss.backward()
                    opt.step()
                if task == "reg":
                    m_ = rmse(yv[va], infer(model, XS[va], ST[va]))
                    better = m_ < best - 1e-5
                else:
                    m_ = roc_auc_score(ybin[va], infer(model, XS[va], ST[va], sig=True))
                    better = m_ > best + 1e-5
                sch.step(m_)
                if better:
                    best, wait = m_, 0
                    best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= PAT:
                        break
            model.load_state_dict(best_state)
            sig = task == "clf"
            oof[va] = infer(model, XS[va], ST[va], sig=sig)
            pred += infer(model, XSe, STe, sig=sig) / len(folds)
        np.savez(os.path.join(PARTS_DIR, f"v15_{task}_s{sd}.npz"), oof=oof, pred=pred)
        accum_oof.append(oof)
        accum_pred.append(pred)
        avg_o = np.mean(accum_oof, 0)
        met = (rmse(yv, avg_o) if task == "reg" else roc_auc_score(ybin, avg_o))
        one = (rmse(yv, oof) if task == "reg" else roc_auc_score(ybin, oof))
        log(f"[{task}] seed={sd} {one:.5f} | {len(accum_oof)}seed 平均 {met:.5f}")
    name = "gru_x2" if task == "reg" else "clf"
    np.savez(os.path.join(out_dir, f"{name}.npz"),
             oof=np.mean(accum_oof, 0), pred=np.mean(accum_pred, 0))
    fin = (rmse(yv, np.mean(accum_oof, 0)) if task == "reg"
           else roc_auc_score(ybin, np.mean(accum_oof, 0)))
    log(f"[{task}] 完成:5 seed 平均 {'OOF' if task == 'reg' else 'AUC'}={fin:.5f}"
        f"(LGB clf AUC 参照 0.90586)-> {out_dir}/{name}.npz")


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    mode = sys.argv[1]
    if mode == "data":
        if not os.path.exists(SEQ2_CACHE):
            build_seq_ext2()
        return
    train(mode, sys.argv[2] if len(sys.argv) > 2 else "0")


if __name__ == "__main__":
    main()
