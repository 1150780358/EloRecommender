# -*- coding: utf-8 -*-
"""v19:周粒度序列 GRU(思路扫荡 3 号,docs/202608/03-v18_思路扫荡.md)。

月度(16 步)有效、交易级(128 步)失效,周粒度(70 步)是未试的中间分辨率。
通道 6:log1p 金额/笔数/商户数/被拒数 + active + 周末金额占比。5-seed 平均。
判据:进池后融合 F31 vs 3.620625(v14 配置),ΔOOF>0.0005。
用法:ELO_SEED=777 python src/v19_weekly.py data|train <dev_id>
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import elo_pipeline as ep
import v13_gru as v13

WK_CACHE = "outputs/wk_tensor.npz"
PARTS_DIR = "outputs/nn_parts"
OUT_DIR = "outputs/base_nn"
W = 70
SEEDS = [777, 1777, 2777, 3777, 4777]
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def build_wk():
    base = pd.read_parquet("data/processed/features.parquet")
    cols = ["card_id", "purchase_date", "purchase_amount", "merchant_id", "authorized_flag"]
    hist = ep.clean_transactions(ep.load_transactions("historical_transactions.csv"))[cols]
    new = ep.clean_transactions(ep.load_transactions("new_merchant_transactions.csv"))[cols]
    tx = pd.concat([hist, new], ignore_index=True)
    del hist, new
    tx["card_id"] = tx["card_id"].astype(str)
    # 每卡以自身最后一笔所在周为 0,向前 70 周(卡内相对时间,与 month_lag 语义一致)
    wk = (tx["purchase_date"].dt.normalize()
          - pd.to_timedelta(tx["purchase_date"].dt.dayofweek, unit="D"))
    tx["wk"] = wk
    last = tx.groupby("card_id", observed=True)["wk"].transform("max")
    tx["wi"] = (W - 1 + ((tx["wk"] - last).dt.days // 7)).astype(np.int32)
    tx = tx[tx["wi"] >= 0]
    tx["wkd_amt"] = np.where(tx["purchase_date"].dt.dayofweek >= 5,
                             np.clip(tx["purchase_amount"], 0, None), 0.0)
    ok = tx[tx["authorized_flag"] == 1]
    g = ok.groupby(["card_id", "wi"], observed=True).agg(
        amt=("purchase_amount", "sum"), cnt=("purchase_amount", "size"),
        mer=("merchant_id", "nunique"), wa=("wkd_amt", "sum"))
    den = (tx[tx["authorized_flag"] == 0]
           .groupby(["card_id", "wi"], observed=True)["purchase_amount"].size().rename("dcnt"))
    g = g.join(den, how="outer")
    log(f"周聚合 {g.shape}")
    piv = {c: g[c].unstack().reindex(columns=range(W)) for c in g.columns}
    cards = piv["amt"].index
    fz = lambda c: piv[c].fillna(0.0).to_numpy(np.float32)
    amt, cnt = fz("amt"), fz("cnt")
    chan = [np.log1p(np.clip(amt, 0, None)), np.log1p(cnt),
            np.log1p(fz("mer")), np.log1p(fz("dcnt")),
            (cnt > 0).astype(np.float32),
            (fz("wa") / np.clip(amt, 1, None)).astype(np.float32)]
    seq = np.stack(chan, axis=2)                     # [N,70,6]
    keep = np.array([1, 1, 1, 1, 0, 0], np.float32)
    idxmap = pd.Index(cards.astype(str))
    out = {}
    zs = np.load(v13.SEQ_CACHE)
    for part, flag in [("tr", 1), ("te", 0)]:
        side = base[base["is_train"] == flag]
        idx = idxmap.get_indexer(side["card_id"].astype(str))
        xs = np.zeros((len(side), W, seq.shape[2]), np.float32)
        hit = idx >= 0
        xs[hit] = seq[idx[hit]]
        out[f"xs_{part}"] = xs
        log(f"{part}: xs={xs.shape}")
    m = out["xs_tr"].reshape(-1, seq.shape[2])
    mu, sd = m.mean(0), m.std(0) + 1e-6
    for p in ("tr", "te"):
        out[f"xs_{p}"] = (out[f"xs_{p}"] - mu * keep) / np.where(keep > 0, sd, 1.0)
    out["st_tr"], out["st_te"] = zs["st_tr"], zs["st_te"]
    np.savez_compressed(WK_CACHE, **out)
    log(f"周序列缓存 {WK_CACHE}: xs_tr={out['xs_tr'].shape}")


def train(dev_id):
    import torch
    import torch.nn as nn
    dev = f"cuda:{dev_id}" if torch.cuda.is_available() else "cpu"
    z = np.load(WK_CACHE)
    xs_tr, st_tr, xs_te, st_te = z["xs_tr"], z["st_tr"], z["xs_te"], z["st_te"]
    base = pd.read_parquet("data/processed/features.parquet")
    y = base[base["is_train"] == 1].reset_index(drop=True)["target"]
    folds = ep.make_folds(y)
    yv = y.to_numpy(np.float32)
    C, S = xs_tr.shape[2], st_tr.shape[1]
    log(f"[wk] train {xs_tr.shape} dev={dev}")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.GRU(C, 96, batch_first=True)
            self.stat = nn.Linear(S, 32)
            self.head = nn.Sequential(
                nn.Linear(96 + 32, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, xs, st):
            _, hn = self.enc(xs)
            return self.head(torch.cat([hn[-1], torch.relu(self.stat(st))], 1)).squeeze(1)

    XS, ST = torch.from_numpy(xs_tr).to(dev), torch.from_numpy(st_tr).to(dev)
    Y = torch.from_numpy(yv).to(dev)
    XSe, STe = torch.from_numpy(xs_te).to(dev), torch.from_numpy(st_te).to(dev)

    def infer(model, xs, st, bs=8192):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(xs[i:i + bs], st[i:i + bs])
                              for i in range(0, len(xs), bs)]).float().cpu().numpy()

    os.makedirs(PARTS_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    BS, MAX_EP, PAT = 1024, 40, 5
    accum_o, accum_p = [], []
    for sd in SEEDS:
        oof, pred = np.zeros(len(yv)), np.zeros(len(xs_te))
        for k, (tr, va) in enumerate(folds):
            torch.manual_seed(sd + k * 101)
            np.random.seed(sd + k * 101)
            model = Net().to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
            lossf = nn.MSELoss()
            tr_t = torch.from_numpy(tr).to(dev)
            best, wait, best_state = 1e9, 0, None
            for _ in range(MAX_EP):
                model.train()
                perm = tr_t[torch.randperm(len(tr_t), device=dev)]
                for i in range(0, len(perm), BS):
                    b = perm[i:i + BS]
                    opt.zero_grad()
                    loss = lossf(model(XS[b], ST[b]), Y[b])
                    loss.backward()
                    opt.step()
                vr = rmse(yv[va], infer(model, XS[va], ST[va]))
                sch.step(vr)
                if vr < best - 1e-5:
                    best, wait = vr, 0
                    best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= PAT:
                        break
            model.load_state_dict(best_state)
            oof[va] = infer(model, XS[va], ST[va])
            pred += infer(model, XSe, STe) / len(folds)
        np.savez(os.path.join(PARTS_DIR, f"wk_s{sd}.npz"), oof=oof, pred=pred)
        accum_o.append(oof)
        accum_p.append(pred)
        log(f"[wk] seed={sd} OOF={rmse(yv, oof):.5f} | {len(accum_o)}seed 平均 {rmse(yv, np.mean(accum_o, 0)):.5f}")
    np.savez(os.path.join(OUT_DIR, "wk.npz"), oof=np.mean(accum_o, 0), pred=np.mean(accum_p, 0))
    log(f"[wk] 完成 5-seed 平均 OOF={rmse(yv, np.mean(accum_o, 0)):.5f} -> {OUT_DIR}/wk.npz")


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    mode = sys.argv[1]
    if mode == "data":
        if not os.path.exists(WK_CACHE):
            build_wk()
    elif mode == "train":
        train(sys.argv[2] if len(sys.argv) > 2 else "1")


if __name__ == "__main__":
    main()
