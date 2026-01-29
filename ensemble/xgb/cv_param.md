| 参数名 | 说明 | 默认值 | 建议设置 |
| :--- | :--- | :--- | :--- |
| **params** | 模型超参数字典（如 eta, max_depth 等） | 无 (必选) | 见模型训练参数 |
| **dtrain** | 训练数据容器 (`xgb.DMatrix`) | 无 (必选) | - |
| **num_boost_round** | 最大迭代次数（树的数量） | 10 | 500 - 2000 (配合早停使用) |
| **nfold** | 交叉验证的折数 | 3 | 5 或 10 |
| **stratified** | 是否进行分层抽样（保持类别比例） | False | 分类任务务必设为 **True** |
| **early_stopping_rounds** | 性能不再提升时提前停止的轮数 | None | 10 - 50 |
| **metrics** | 评估指标 (string 或 list) | () | 'auc', 'logloss', 'error', 'rmse' |
| **seed** | 随机种子，保证 CV 结果可复现 | 0 | 42 或 任意固定整数 |
| **as_pandas** | 返回结果是否为 Pandas DataFrame | True | **True** (方便分析) |
| **verbose_eval** | 进度打印频率 (bool 或 int) | None | 10 或 True |
| **show_stdv** | 是否在进度中打印标准差 | True | True |
| **shuffle** | 在划分 fold 前是否打乱数据 | True | True |

重要参数：
stratified (分类任务核心):
    在处理分类问题（尤其是类别不平衡时），必须设为 True。这确保了每一折数据中正负样本的比例与全量数据一致，避免评估偏差。

num_boost_round early_stopping_rounds:

    不要纠结 num_boost_round 到底设多少。通常设一个较大的值（如 1000），然后配置 early_stopping_rounds=50。这样模型会在验证集指标不再下降时自动停止，并返回此时的轮数。

metrics:
    如果你在 params 里定义了 eval_metric，cv 会默认使用它。但你也可以在这里覆盖它，例如 metrics=['auc', 'logloss'] 可以同时监控多个指标。

nfold:

    5 折: 最常用，兼顾速度与评估的稳定性。
    10 折: 数据量较小时（如 < 5000 条），建议使用 10 折以获得更可靠的评估。


cv_results 返回的是各轮迭代的均值和标准差。通常我们使用 test-[metric]-mean 来选择最佳轮数