# outputs/ 产物清单

> 本目录是**脚本管理的工作目录**:所有路径被 `src/` 下脚本硬编码引用,同名文件会被复跑**覆盖写**。
> 请勿手工移动/重命名;历史成绩以 `docs/submissions.md` 流水为准,不依赖本目录留档。
> 除本 README、`*.json` 报告、`*.txt` 描述与 `feature_importance*.csv` 外,其余(csv/npz/png/logs/base*/seed*)均被 `.gitignore` 排除,仅存本地。

## 基模型产物(热区,谨慎操作)

| 路径 | 产出者 | 消费者 | 说明 |
|---|---|---|---|
| `base/{name}.npz` | `v5_base.py` | `v5_fusion.py` / `v6_te.py` | 统一基模型产物协议:每个 npz 含 `oof` + `test` 两数组;name ∈ lgb/xgb/cat/hub/clean/clf/mlp/mlp2/et/clf_* |
| `base_ext/*.npz` | `v5_train_ext.py` | `v5_fusion.py`(前缀 `x_`) | v5 增量特征集(149 列被拒画像+近窗)基模型 |
| `base_te/*.npz` | `v6_te.py` | `v5_fusion.py`(前缀 `t_`) | **v6 TE-v1 特征集基模型(线上验证 Private 3.60635,已冻结)**;泄漏版 v2/v2_tm/v3 产物已于 2026-07-31 清除 |
| `base_te_v4/*.npz` | `v6_te.py` | 待验证 | TE-v4(v1+众数键)基模型,OOF 改善幅度正常,验证中 |
| `base_mte/` + `merchant_te.npz` | `v6_merchant_te.py` | — | 商户侧 TE,已证伪(OOF 不动),收官归档时可删 |
| `te_features.npz` / `te_features_v4.npz` | `v6_te.py` | `v6_te.py` | TE-v1 / v4 特征缓存;v2/v2_tm/v3 缓存已随泄漏清理删除 |

## 提交文件(每轮全量复跑会覆盖)

| 文件 | 产出者 | 说明 |
|---|---|---|
| `submission_{lgb,xgb,cat,stack,blend}.csv` | `elo_pipeline.py` | 单模与一层融合;stack 为 v1–v4 主力 |
| `submission_clean_pp.csv` | `exp1_clean_postprocess.py` | clean 模型 + outlier 概率软融合 |
| `submission_seedavg_{clean_pp,stack}.csv` | `avg_seeds.py` | v4 3-seed 平均(clean_pp 版 Private 3.60759) |
| `submission_v5_fusion.csv` | `v5_fusion.py` | 二层融合终选(F10 版 Private 3.60753,当前最优;v6 复跑会覆盖) |
| `submission_v6_te.csv` | `v6_te.py` | v6 TE-v1 F20 提交存档(Private 3.60635,当前最优) |

## 训练与实验报告(小文件,git 跟踪)

| 文件 | 产出者 | 说明 |
|---|---|---|
| `cv_summary.json` | `elo_pipeline.py` | 十折逐折 RMSE / AUC / stacking 系数 |
| `best_params.json` | `exp3_tune.py` | Optuna 终选参数;`elo_pipeline.py` tuned 模式读取 |
| `exp1_desc.txt` / `exp1_report.json` | `exp1_clean_postprocess.py` | 后处理择优描述(战役脚本取 desc 作提交备注)与扫描报告 |
| `v5_fusion_desc.txt` / `v5_fusion_report.json` | `v5_fusion.py` | 融合方案对比全表与终选描述 |
| `feature_importance.csv` / `.png` | `elo_pipeline.py` | 主特征集 gain 重要性(png 不入库) |
| `feature_importance_ext.csv` | `v5_train_ext.py` | 增量特征集重要性 |
| `private_lb_top200.json` | Kaggle API(一次性取回) | **私榜真实分数标尺**,等效名次换算依据,勿删 |
| `oof_predictions.csv` / `test_clf_prob.csv` | `elo_pipeline.py` | 六路 OOF 与 outlier 概率;已由 `v5_base.py migrate` 迁入 `base/*.npz`,保留作原始凭证 |

## 冻结区(实验产物,仅历史价值)

| 路径 | 体积 | 说明 |
|---|---|---|
| `seed2019/` `seed42/` `seed777/` | 3×45M | v4 3-seed 战役逐 seed 产物,`avg_seeds.py` 的输入 |
| `logs/` | <1M | 全部 nohup 运行日志(campaign / v5_* / v6_*) |

## 战役收官后的归档清单(待所有实验停止后执行)

1. `seed*/` 三目录 `tar zcf archive/v4_seeds.tgz` 后删除原目录(如需重跑 `avg_seeds.py` 先解包);
2. 单模提交文件(lgb/xgb/cat/blend)无保留价值,可删——线上成绩全部记录在 `docs/submissions.md`;
3. `logs/` 按版本打包归档;
4. 归档动作前先 `find outputs -mmin -120` 确认无实验在写。
