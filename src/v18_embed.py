# -*- coding: utf-8 -*-
"""v18:GRU hidden embedding → LGB(思路扫荡 2 号,docs/202608/03-v18_思路扫荡.md)。

v16 证明手工统计量无法把序列信息带回树;本脚本改用**学出来的表征**:
10 通道 gru_x 架构逐折训练,导出 96 维最终 hidden(OOF=折模型出 val;test=10 折平均),
喂给 lgb(sel+TE+td+fm+hid96)。判据 vs outputs/base_fm/lgb.npz(3.63246)>0.0005。
用法:ELO_SEED=777 python src/v18_embed.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import elo_pipeline as ep
import v11_formula as v11

SEQX = "outputs/seq_tensor_ext.npz"
REF_LGB = "outputs/base_fm/lgb.npz"
EMB_CACHE = "outputs/gru_hidden.npz"
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def export_hidden(folds, yv):
    import torch
    import torch.nn as nn
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    z = np.load(SEQX)
    xs_tr, st_tr, xs_te, st_te = z["xs_tr"], z["st_tr"], z["xs_te"], z["st_te"]
    C, S = xs_tr.shape[2], st_tr.shape[1]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.GRU(C, 96, num_layers=2, batch_first=True, dropout=0.1)
            self.stat = nn.Linear(S, 32)
            self.head = nn.Sequential(
                nn.Linear(96 + 32, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, xs, st, want_hidden=False):
            _, hn = self.enc(xs)
            h = hn[-1]
            if want_hidden:
                return h
            return self.head(torch.cat([h, torch.relu(self.stat(st))], 1)).squeeze(1)

    XS, ST = torch.from_numpy(xs_tr).to(dev), torch.from_numpy(st_tr).to(dev)
    Y = torch.from_numpy(yv).to(dev)
    XSe, STe = torch.from_numpy(xs_te).to(dev), torch.from_numpy(st_te).to(dev)

    def hidden(model, xs, bs=8192):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(xs[i:i + bs], None, want_hidden=True)
                              for i in range(0, len(xs), bs)]).float().cpu().numpy()

    def infer(model, xs, st, bs=8192):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(xs[i:i + bs], st[i:i + bs])
                              for i in range(0, len(xs), bs)]).float().cpu().numpy()

    hid_tr = np.zeros((len(xs_tr), 96), np.float32)
    hid_te = np.zeros((len(xs_te), 96), np.float32)
    BS, MAX_EP, PAT = 1024, 40, 5
    for k, (tr, va) in enumerate(folds):
        torch.manual_seed(777 + k * 101)
        np.random.seed(777 + k * 101)
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
        hid_tr[va] = hidden(model, XS[va])
        hid_te += hidden(model, XSe) / len(folds)
        log(f"[emb] fold{k + 1} val_rmse={best:.5f} hidden 导出")
    np.savez_compressed(EMB_CACHE, tr=hid_tr, te=hid_te)
    log(f"hidden 缓存 {EMB_CACHE}")
    return hid_tr, hid_te


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    base = pd.read_parquet("data/processed/features.parquet")
    base = base.merge(v11.hist_lag_amt(), on="card_id", how="left")
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    yv = y.to_numpy(np.float32)
    folds = ep.make_folds(y)
    if os.path.exists(EMB_CACHE):
        zz = np.load(EMB_CACHE)
        hid_tr, hid_te = zz["tr"], zz["te"]
    else:
        hid_tr, hid_te = export_hidden(folds, yv)
    fm_tr, fm_te = v11.formula_block(train), v11.formula_block(test)
    imp = pd.read_csv("outputs/feature_importance.csv")
    sel = [c for c in imp[imp["gain"] > 0].head(ep.CONFIG["TOP_K"])["feature"] if c in train.columns]
    z = np.load("outputs/te_features.npz", allow_pickle=True)
    te_names = [str(x) for x in z["names"]]
    td = pd.read_parquet("outputs/td_features.parquet")
    hn = [f"hid_{i}" for i in range(hid_tr.shape[1])]

    def assemble(side, zte, fm, hid):
        m1 = side[["card_id"]].merge(td, on="card_id", how="left").drop(columns="card_id")
        return pd.concat([side[sel].reset_index(drop=True), pd.DataFrame(zte, columns=te_names),
                          m1.astype(np.float32), fm.reset_index(drop=True),
                          pd.DataFrame(hid, columns=hn)], axis=1)

    X = assemble(train, z["tr"], fm_tr, hid_tr)
    X_test = assemble(test, z["te"], fm_te, hid_te)
    log(f"X={X.shape}(含 hid {len(hn)})")
    oof, pred, _, gain = ep.cv_lightgbm(X, y, X_test, folds, ep.LGB_PARAMS, "lgb+hid")
    os.makedirs("outputs/base_hb", exist_ok=True)
    np.savez("outputs/base_hb/lgb.npz", oof=oof, pred=pred)
    ref = rmse(y, np.load(REF_LGB)["oof"])
    s = rmse(y, oof)
    d = ref - s
    log(f"判据[v18 hidden→LGB]:OOF={s:.5f} vs fm 基线 {ref:.5f} → 改善 {d:+.5f} "
        f"{'✅ 通过' if d > 0.0005 else '❌ 不足'}")
    g2 = pd.DataFrame({"feature": X.columns, "gain": gain}).sort_values("gain", ascending=False)
    log(f"hid 列进入 gain 前 50:{int(g2.head(50)['feature'].str.startswith('hid_').sum())};"
        f"前 5 hid:\n" + g2[g2["feature"].str.startswith("hid_")].head(5).to_string(index=False))
    if d <= 0.0005:
        sys.exit(3)


if __name__ == "__main__":
    main()
