# -*- coding: utf-8 -*-
"""实验③:Optuna(TPE)复搜三模型超参并用终选参数十折复训(直接复用主管线)。"""
import json

import elo_pipeline as ep

ep.CONFIG["TUNE"] = True
ep.CONFIG["TUNE_TRIALS"] = 40
ep.main()  # 特征已缓存 → 直接筛选+调参+十折复训+融合+保存全部产物

json.dump({"lgb_best_trial": {k: v for k, v in ep.LGB_PARAMS.items()
                              if k not in ("verbosity", "num_threads", "seed", "metric",
                                           "objective", "boosting")},
           "xgb": ep.XGB_PARAMS, "cat": ep.CAT_PARAMS},
          open("outputs/best_params.json", "w"), indent=2, default=str)
print("[exp3] 终选参数已保存 outputs/best_params.json", flush=True)
