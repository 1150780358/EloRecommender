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

**当前最优:v14 F31 全池融合(NN 族 ×3 + FM + TD + TE + v4 池,31 元特征,BayesianRidge),Private 3.59764 → 私榜等效第 7 名 / 4111(top 0.17%,金牌区)。**
较首日 stack 3.61276(~210 名)累计提升 -0.0151;距第 6 名(3.59707)差 0.00057,第 5 名(3.59422)差 0.0034,第 1 名(3.57875)差 0.0189。
> 2026-08-03 v14 战役:NN 族纵深三重奏 —— 5-seed 平均(−0.009)、10 通道 ext(**被拒交易逐月序列**,gru_x 3.67436 单 seed 即打平 gru 5-seed 平均)、Transformer 异构。新 F31(NN×3)ΔOOF −0.00100 → ΔPrivate −0.00104,**转化率 104%**;12→10→7 一日双跳。见 `docs/202608/03-v14_NN族纵深.md`。
> 2026-08-03 v15/v16 收官判据:v15 NN 通道扩展(13ch 持平)与 clf 头 NN 化(ens AUC 0.9078 但派生链/裸列均不转化)三用法证伪;v16 dec 时间结构 + merchants 增量判据 **−0.00133**(高 gain + 负 OOF,弱信息列过拟合)。冲第 6 弹药出清,**项目收官定位第 7 名**。见 `docs/202608/03-v15_通道扩展与clf头.md`、`03-v16_dec时间结构.md`。
> 2026-08-03 v17 终局判据:交易级序列 GRU(近 128 笔 token 化,5-seed 3.72934,异构性全场最强 vs 树 0.698)融合 **+0.00015 恶化** → 移出池。「弱而不同」有精度下限(v13 弱 0.06 可用,v17 弱 0.10 反稀释)。高上限候选全部出清,**项目终局定格第 7 名**。见 `docs/202608/03-v17_交易级序列.md`。
> 2026-08-03 v18-v20 思路扫荡(用户推动的结构性遗漏三路):① 树多 seed 平均 +0.00001(降噪与融合是同一方差机制,不叠加,xgb 单模虽创 3.63072 新高)② GRU hidden→LGB −0.00597(载体不可交换双向闭环)③ 周粒度序列 +0.00011(月度是唯一分辨率甜点)。**三路全灭,第 7 名确认为收敛点,系统内继续提升仅剩线上赌博。**见 `docs/202608/03-v18_思路扫荡.md`。
> 2026-08-03 v21 榜单二次考古(用户推动):检索出 6 个未读帖(11th/18th/32nd/xentropy 帖等)。两个最高先验候选均拦截:xentropy 单模 3.63147 创 lgb 系新高但与 L2 版相关 0.9922,融合 +0.00004(**深池吸收定律**:浅池时代榜单配方对 32 成员深池失效);seq-emb(PPMI-SVD)−0.00065。独立收获:11th 实盘证 topN 替换 Private 零效、18th 实盘证后处理损失 8 名、11th/32nd NN 全灭衬托我们 NN 落地优势。**榜单公开信息全部出清。**见 `docs/202608/03-v21_榜单二次考古.md`。
> 2026-08-03 v13 战役:序列端到端 GRU(16 步 × 7 通道月度序列,静态分支刻意瘦)。单模 OOF 3.69455 弱于树 0.06,但与树预测相关仅 0.77(树际 0.95+),**融合增益 ΔOOF −0.00179**(F31 vs F29)→ ΔPrivate −0.00126,转化率 70%。**异构性**是继「新信息」后第二条被验证的路径。见 `docs/202608/03-v13_序列端到端.md`。
> 2026-08-03 v12 判据:fm 族 retention 纵深(交叉 retention/重访/合成分子,12 列)中性证伪(+0.00007);数据实证 new 表无重访(lag1∩lag2 商户 = 0)→ 评估窗回头交易不可观测,公式逆向路线收益到顶。见 `docs/202608/03-v12_fm纵深.md`。
> 2026-07-31 v11 战役:target 逆向考古坐实 **target = log2(x+1e-10)**(格点=简单分数、哨兵=log2(1e-10)),16 列公式形状 log2 比值特征(ε 与哨兵同构)单模 +0.0007、clf AUC 0.90586 新高,F29 融合 ΔOOF −0.00114 → ΔPrivate −0.00109(协议一致口径,**转化率 96%**)。折外无依赖家族第三次高转化。见 `docs/202607/31-v11_target逆向.md`。
> 2026-07-31 v7 战役:21st 交易粒度时间差分布(a2p/p2r/p2now,30 列)单模 ΔOOF −0.0065(超警报线,三重审计通过:对抗验证 AUC 0.497 / 十折全改善 / 机制上无泄漏通道),融合 ΔOOF −0.00481 → ΔPrivate −0.00571,**转化率 119%** —— 折外无依赖的行为特征转化率高于 TE(45%),再次验证「新信息」路径。弱正则参数对照证伪(OOF +0.00486)。见 `docs/202607/31-v7_21st收尾与Top1复现.md`。
> ⚠️ v7/v8 的 OOF 数字含 2019/777 折协议错配的乐观偏置(见 `31-v7-seed错配警告.md`),Private 为线上实测不受影响;777 规范复跑后以新 OOF 为基线。
> 2026-07-31 v9 战役:E1(16th 稳健 clean)融合层证伪;E2(5th 分段建模)离线 −0.00167 但线上 +0.00013 **不转化**(结构层收益第三次证伪);777 协议对照提交 3.60103(比混折原版差 0.0004,协议差异属线上噪声带,修协议是为判据可信而非分数)。**当前最优维持 v7 F23 原版,项目就此收官定位第 16 名。**见 `docs/202607/31-v9_剩余候选验证.md`。
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
| **v7 F23 全池(TD 时间差分布) | 3.60064 | 16 | 金(top 0.4%) |
| v9 F27(+5th 分段,777 协议) | 3.60077 | 18 | 金线上,未超 v7 |
| v7 F23 777 协议对照 | 3.60103 | ~20 | 参照:协议差异属噪声带 |
| v11 F29 全池(FM 公式特征) | 3.59994 | 12 | 金(top 0.29%) |
| v13 F31 全池(+序列 GRU) | 3.59868 | 10 | 金(top 0.24%),Top10 达成 |
| **v14 F31 全池(NN 族 ×3,当前最优)** | **3.59764** | **7** | **金(top 0.17%)** |

~~金牌需 ≤3.60078~~ 已达成(2026-07-31);~~Top10 需 <3.59904~~ 已达成(2026-08-03,v13);当前等效第 7。下一参照:第 6 名 3.59707(差 0.00057)、第 5 名 3.59422(差 0.0034)、第 1 名 3.57875(差 0.0189)。路径沉淀:**新信息**(v6 45% / v7 119% / v11 96%)与**异构性**(v13 70% / v14 104%)双路径确立;被拒交易序列 = 新信息 × 异构载体的乘积样本;模型层/融合层/结构层调优三路皆零。

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
| 2026-07-31 07:23 | submission_v9.csv | v9 F27 (777-protocol): F23 pool + seg (5th-style two-segment modeling by calibrated clf prob thr 0.015) + clean2, BayesianRidge, OOF 3.62287 | 3.67825 | 3.60077 |
| 2026-07-31 07:25 | submission_v7_td_s777.csv | v7 F23 777-protocol rerun (same config as 3.60064 submission, fold-protocol fix only), OOF 3.62454 | 3.67898 | 3.60103 |
| 2026-07-31 09:57 | submission_v11_fm.csv | v11 F29 (777): F23 pool + FM formula-shape suite (target=log2(x+1e-10) archaeology, 16 log2-ratio cols, ep-sentinel-aligned) 5 base models, BayesianRidge, OOF 3.62340 | 3.68003 | 3.59994 |
| 2026-08-03 01:33 | submission_v13_nn.csv | v13 F31 (777): F29 pool + seq GRU (16x7 monthly, pred-corr 0.77 vs trees), fusion dOOF -0.00179 | 3.67867 | 3.59868 |
| 2026-08-03 03:14 | submission_v14_nn3.csv | v14 F31 (777): F29 pool + NN family x3 (gru/gru_x-10ch-denied/trf, 5-seed avg each), fusion dOOF -0.00100 | 3.67710 | 3.59764 |
