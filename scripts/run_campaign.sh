#!/bin/bash
# Elo 自动实验战役:脱离终端/会话独立运行(nohup+setsid 启动)
# 阶段② 观察月特征全套重训 → ① clean+后处理 → ③ Optuna 复搜复训 → ①b tuned 复跑
# 每阶段自动提交 Kaggle 并回填 docs/submissions.md;全程日志 outputs/logs/campaign.log
cd /data/user/cwz/EloRecommender || exit 1
PY=/data/guest/anaconda3/bin/python
exec >> outputs/logs/campaign.log 2>&1

echo "===== CAMPAIGN START $(date '+%F %T') ====="

echo "----- [stage2] ref-month 特征重建 + 全套十折重训 -----"
rm -f data/processed/features.parquet
$PY src/elo_pipeline.py || { echo "STAGE2_FAILED $(date '+%F %T')"; exit 1; }
$PY src/submit_and_log.py outputs/submission_stack.csv "v2 ref-month features, ridge stacking"

echo "----- [stage1] clean 模型 + 后处理(基于 v2 工件)-----"
if $PY src/exp1_clean_postprocess.py; then
    $PY src/submit_and_log.py outputs/submission_clean_pp.csv "v2 $(cat outputs/exp1_desc.txt)"
else
    echo "STAGE1_FAILED $(date '+%F %T')"
fi

echo "----- [stage3] Optuna TPE 40x3 复搜 + 终选参数复训 -----"
$PY src/exp3_tune.py || { echo "STAGE3_FAILED $(date '+%F %T')"; exit 1; }
$PY src/submit_and_log.py outputs/submission_stack.csv "v3 tuned params, ridge stacking"

echo "----- [stage1b] clean+后处理复跑(tuned 工件)-----"
if $PY src/exp1_clean_postprocess.py --tuned; then
    $PY src/submit_and_log.py outputs/submission_clean_pp.csv "v3 $(cat outputs/exp1_desc.txt)"
else
    echo "STAGE1B_FAILED $(date '+%F %T')"
fi

echo "===== CAMPAIGN DONE $(date '+%F %T') ====="
