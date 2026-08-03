import numpy as np, pandas as pd
import v5_fusion as vf, elo_pipeline as ep

bases = vf.load_bases()
zc = np.load("outputs/base_nn_clf/clf.npz")
bases["nn_pclf"] = (zc["oof"], zc["pred"])
base = pd.read_parquet("data/processed/features.parquet")
train = base[base["is_train"] == 1].reset_index(drop=True)
y = train["target"]
ybin = (y < -30).astype(int).to_numpy()
folds = ep.make_folds(y)
REG = [k for k in ("lgb", "xgb", "cat", "hub") if k in bases]
T = [k for k in ("t_lgb", "t_xgb", "t_cat", "t_hub") if k in bases]
D = [k for k in ("d_lgb", "d_xgb", "d_cat", "d_hub") if k in bases]
F = [k for k in ("f_lgb", "f_xgb", "f_cat", "f_hub") if k in bases]
N = sorted(k for k in bases if k.startswith("n_"))
allf4 = (REG + T + D + F + ["t_clf", "t_clean", "d_clf", "d_clean",
         "f_clf", "f_clean", "p_cal", "ev", "p_cal_x_clean"] + N)
r0, _, _ = vf.evaluate(allf4, "bayes", bases, y, ybin, folds, p_src="f_clf", clean_src="f_clean")
r1, _, pt = vf.evaluate(allf4 + ["nn_pclf"], "bayes", bases, y, ybin, folds, p_src="f_clf", clean_src="f_clean")
print(f"F31 复算 OOF={r0:.5f}")
print(f"F34(+nn_pclf 裸列)OOF={r1:.5f}  Δ={r0 - r1:+.5f} {'✅' if r0 - r1 > 0.0005 else '❌ 不足'}")
if r0 - r1 > 0.0005:
    sub = pd.read_csv("data/raw/sample_submission.csv")
    sub["target"] = pt
    sub.to_csv("outputs/submission_v15b.csv", index=False)
    print("已保存 outputs/submission_v15b.csv")
