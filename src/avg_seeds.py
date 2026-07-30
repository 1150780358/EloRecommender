# -*- coding: utf-8 -*-
"""多 seed 预测平均:合并 outputs/seed*/ 下各 seed 的提交,行序校验后取均值。"""
import glob

import numpy as np
import pandas as pd

for name, out in [("submission_clean_pp.csv", "outputs/submission_seedavg_clean_pp.csv"),
                  ("submission_stack.csv", "outputs/submission_seedavg_stack.csv")]:
    paths = sorted(glob.glob(f"outputs/seed*/{name}"))
    dfs = [pd.read_csv(p) for p in paths]
    assert len(dfs) >= 2, f"{name} 少于 2 个 seed"
    for d in dfs[1:]:
        assert (d["card_id"].values == dfs[0]["card_id"].values).all(), "card_id 行序不一致"
    avg = dfs[0][["card_id"]].copy()
    avg["target"] = np.mean([d["target"].values for d in dfs], axis=0)
    avg.to_csv(out, index=False)
    print(f"[avg] {out} <- {len(paths)} seeds: {paths}", flush=True)
