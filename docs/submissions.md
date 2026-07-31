# Elo Merchant Category Recommendation — 刷榜记录

- **赛题**:elo-merchant-category-recommendation(2019-02-26 已截止,Late Submission 不进官方榜)
- **基准信息**:总队伍 4111;奖牌线 金≈18 / 银≈206 / 铜≈411;**私榜第 1 名 3.57875**(2026-07-30 经 Kaggle API 核实,此前记录的 3.58657 有误);公榜第 1 名 3.61285(私榜大洗牌)
- **私榜真实锚点**(`outputs/private_lb_top200.json`,API `competitions/.../leaderboard/view?pageSize=500` 可取 top200):

  | 名次 | 1 | 10 | **18(金线)** | 30 | 50 | 69 | 100 | 150 | 200 |
  |---|---|---|---|---|---|---|---|---|---|
  | Private | 3.57875 | 3.59904 | **3.60078** | 3.60319 | 3.60543 | 3.60759 | 3.60932 | 3.61070 | 3.61167 |
- **CV↔LB 换算**:分层十折 OOF → Private 偏移 ≈ **-0.031**(以此离线推算线上收益,节省提交)
- **提交纪律**:每日额度约 100 次;新实验先看 OOF,仅最终候选上线验证;每次提交必须回填本表
- **⚠️ 泄漏警报线(2026-07-30 v6-v2 事故后新增)**:单一改动若使**单模 OOF 改善超过 0.003**,一律先视为泄漏嫌疑,不得直接上线 —— 参照系是 v1→v4 全部工程改进累计仅 0.0052。实测 TE-v2 单模 OOF 改善 0.0077(6 倍于 v1),线上 Private 反而恶化 0.0059。

| # | 时间(UTC) | 提交文件 | 方案说明 | OOF | Public | Private | Private 等效排名(估) |
|---|---|---|---|---|---|---|---|
| 1 | 2026-07-29 06:23 | submission_stack.csv | Ridge Stacking(LGB+XGB+CAT+outlier clf),分层 10 折,seed=2019,190 特征 | 3.64362 | 3.69332 | **3.61276** | ~205–215 / 4111(银牌线边缘) |
| 2 | 2026-07-29 06:36 | submission_blend.csv | SLSQP 非负加权(cat .468 / xgb .303 / lgb .230) | 3.64382 | 3.69598 | 3.61316 | 略逊于 #1 |

**当前最优:v7 F23 全池融合(TD 时间差分布 ×5 基模型 + TE 池 + v4 池,BayesianRidge 二层),Private 3.60064 → 私榜等效第 16 名 / 4111(top 0.4%,金牌区)**。
较首日 stack 3.61276(~210 名)累计提升 -0.0121;**已越过金牌线(18 名,3.60078)0.00014**。
> 2026-07-31 v7 战役:21st 交易粒度时间差分布(a2p/p2r/p2now,30 列)单模 ΔOOF −0.0065(超警报线,三重审计通过:对抗验证 AUC 0.497 / 十折全改善 / 机制上无泄漏通道),融合 ΔOOF −0.00481 → ΔPrivate −0.00571,**转化率 119%** —— 折外无依赖的行为特征转化率高于 TE(45%),再次验证「新信息」路径。弱正则参数对照证伪(OOF +0.00486)。见 `docs/202607/31-v7_21st收尾与Top1复现.md`。
> ⚠️ v7/v8 的 OOF 数字含 2019/777 折协议错配的乐观偏置(见 `31-v7-seed错配警告.md`),Private 为线上实测不受影响;777 规范复跑后以新 OOF 为基线。
> 2026-07-30 用真实私榜 top200 重新定位:此前"铜牌线附近"的估计严重低估,实际早已进入银牌区。
> v5 战役(融合层重构 / 异构模型 / 新特征族)三个方向全部证伪,见 `docs/202607/30-v5_fusion_heterogeneous.md`。
> v6 战役移植 21st 的**折外 outlier 率目标编码**,是首个真正转化到线上的改动(ΔOOF −0.00263 → ΔPrivate −0.00118,转化率 45%,对比 v5 的 2%),见 `docs/202607/30-v6_target_encoding.md`。

## 结论沉淀

1. Stacking > blend 在 OOF/Public/Private 三处排序一致 → 验证方案可信,离线择优即可;
2. 公榜排名(1493/4111)无参考价值,一切以 Private/OOF 为准;
3. **战役复盘(2026-07-29)**:后处理是最大赢家(clean+概率融合 α=0.5,OOF -0.002 → Private -0.0031);观察月特征基本中性(Private -0.00004);Optuna 复搜未超过预设参(验证"改善<0.002 保留预设"纪律);topN 硬替换全面差于概率软融合;
4. 后续方向:异构基学习器(NN/FFM)、多 seed 平均、clean 模型也纳入 stacking 一层。

## 私榜等效排名一览(2026-07-30 核定)

| 版本 | Private | 等效名次 / 4111 | 等级 |
|---|---|---|---|
| v1 stack | 3.61276 | ~210 | 银线边缘 |
| v2 clean+pp | 3.60974 | 114 | 银 |
| v3 clean+pp | 3.60965 | 112 | 银 |
| v4 stack | 3.60984 | 116 | 银 |
| v4 clean+pp | 3.60772 | 72 | 银 |
| v4 3-seed avg clean+pp | 3.60759 | 69 | 银(top 1.7%) |
| v5 F10 融合 | 3.60753 | 69 | 银(top 1.7%) |
| v6 F20 全池(TE 目标编码) | 3.60635 | 59 | 银(top 1.4%) |
| **v7 F23 全池(TD 时间差分布,当前最优)** | **3.60064** | **16** | **金(top 0.4%)** |

~~金牌需 ≤3.60078~~ **已达成(2026-07-31)**。下一参照:第 10 名 3.59904(差 0.0016)、第 1 名 3.57875(差 0.0219)。v5 已证明模型层与融合层边际收益为零;v6(TE,转化率 45%)与 v7(时间差分布,转化率 119%)确认**新信息**是唯一有效路径,且折外无依赖特征的转化率显著高于折依赖的编码类特征。

## 自动实验提交流水(脚本自动回填,追加在本表末尾)

| 时间(UTC) | 提交文件 | 方案说明 | Public | Private |
|---|---|---|---|---|
| 2026-07-29 09:16 | submission_stack.csv | v2 ref-month features, ridge stacking | 3.69293 | 3.61272 |
| 2026-07-29 09:21 | submission_clean_pp.csv | v2 clean+pp blend-0.5, OOF 3.64195 | 3.69253 | 3.60974 |
| 2026-07-29 11:18 | submission_stack.csv | v3 tuned params, ridge stacking | 3.69303 | 3.61250 |
| 2026-07-29 11:22 | submission_clean_pp.csv | v3 clean+pp blend-0.5, OOF 3.64228 | 3.69274 | 3.60965 |
| 2026-07-29 13:00 | submission_stack.csv | v4 opensource-port feats, 4-model stack | 3.68599 | 3.60984 |
| 2026-07-29 13:06 | submission_clean_pp.csv | v4 clean+pp blend-0.5, OOF 3.63670 | 3.68618 | 3.60772 |
| 2026-07-29 14:40 | submission_seedavg_clean_pp.csv | v4 3-seed avg clean+pp | 3.68651 | **3.60759** |
| 2026-07-29 14:40 | submission_seedavg_stack.csv | v4 3-seed avg stack | 3.68627 | 3.60973 |
| 2026-07-30 02:58 | submission_v5_fusion.csv | v5 fusion F10 (7 base models incl MLP/MLP2/ET + clean + ev, BayesianRidge 2nd layer), no-hub, OOF 3.63474 | 3.68770 | 3.60753 |
| 2026-07-30 06:45 | submission_v6_te.csv | v6 F20 full-pool: TE(21st out-of-fold outlier-rate target encoding, 36 cols) x 4 GBDT + v4-feature 7 models + MLP/ET, BayesianRidge 2nd layer, OOF 3.63192 | 3.68423 | 3.60635 |
| 2026-07-30 08:45 | submission_v6_te_v2.csv | v6 F20 TE-v2: card-side OOF target encoding 69 cols (11 low-card keys + 3 high-card mode keys) x 4 GBDT + v4 7 models + MLP/ET, BayesianRidge, OOF 3.62760 | 3.68655 | 3.61226 |
| 2026-07-31 02:27 | submission_v7_td.csv | v7 F23 full-pool TD+TE+orig: 30 timediff-dist cols (21st a2p/p2r/p2now, leak-audited) x 5 td base models + TE pool + v4 pool, BayesianRidge, OOF 3.62711 | 3.67880 | 3.60064 |
