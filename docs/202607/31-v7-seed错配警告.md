# ⚠️ v7 时间差实验 seed 错配警告(2026-07-31,清理会话留)

## 事件

07-31 09:36 启动的 v7 链条 `(v7_timediff.py rest && ELO_FUSION_OUT=… v5_fusion.py)` **未设 `ELO_SEED=777`**,
折切分落到 `elo_pipeline.py` 默认 **seed 2019**。而现存全部产物(`base*/`、`te_features.npz`)均为 **777 折**
(见审计文档 S5 折指纹结论)——审计标记的"换 seed 静默复用旧折编码"之雷已被触发。

## 影响

1. **base_td/ 全部 npz(lgb / lgb_weakreg / xgb / cat …)是 2019 折 OOF**,与 777 折基线
   `base_te/lgb.npz 3.64170` 不可比;且 2019 折 CV 叠 777 折 TE 编码 → 验证折同伴标签漏入编码,
   OOF 含乐观偏置。**v7 的 go/no-go 判据(改善 >0.0005)在此协议下无效。**
2. 链尾 fusion 未设 seed 且会把 `d_*`(2019 折)混入 777 折元特征池,并**覆盖**
   `v5_fusion_report.json` / `v5_fusion_desc.txt` / `submission_v5_fusion.csv` —— 其输出表与历史全表不可比。

## 偏置实证(同一输入,仅二层折 seed 不同)

| 二层折 | F20 全池 bayes OOF |
|---|---|
| 777(规范,折与基模型一致) | **3.63192**(产出与线上 3.60635 提交文件 **md5 一致**) |
| 2019(错配) | 3.63145(虚好 0.0005,方向与审计 S5 预测一致) |

## 处置建议

- v7 数字**先不采信、不提交**;`ELO_SEED=777` 重跑 `base_td` 全套与 fusion 后再评估;
- 任何跨产物比较前按审计 P4 先做折指纹断言;缓存/产物名尽快加 seed 后缀。

## 本会话已完成

- 泄漏产物清除:`base_te_v2/`、`base_te_v2_tm/`、`base_te_v3/`、`te_features_v2(.tm)/v3.npz`、
  `submission_v6_te_v2.csv`(删前 OOF RMSE 对账,v1/v4 保留);
- `v6_merchant_te.py:154` 悬空引用(→ 已删的 v2 缓存)改指 v1,并注明原 P5 证伪基线是 v2;
- 报告三件套以 `ELO_SEED=777` + 排除 base_td 的临时副本复跑恢复规范版(F20 3.63192)。
