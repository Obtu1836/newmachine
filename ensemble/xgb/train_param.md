| 参数名 | 说明 | 默认值 | 建议设置 |
| :--- | :--- | :--- | :--- |
| **params** | 模型超参数字典 | 无 (必选) | 见之前的 `train_param.md` |
| **dtrain** | 训练数据 | 无 (必选) | `xgb.DMatrix` 对象 |
| **num_boost_round** | 最大迭代次数（树的数量） | 10 | **500 - 2000** (配合早停) |
| **evals** | 训练期间监控的数据集列表 | None | `[(dtrain, 'train'), (dval, 'val')]` |
| **early_stopping_rounds** | 性能不再提升时提前停止的轮数 | None | **50** |
| **verbose_eval** | 打印训练日志的频率 | True | **10** 或 **50** (避免日志刷屏) |
| **evals_result** | 字典，用于存储每个周期的评估结果 | None | 传入一个空字典 `{}` 以供后续绘图 |
| **obj** | 自定义目标函数 | None | 仅在需要特殊损失函数时设置 |
| **feval** / **custom_metric** | 自定义评估指标函数 | None | 仅在评估指标无法通过字符串定义时设置 |
| **xgb_model** | 用于继续训练的先前模型 | None | 传入路径或 Booster 对象实现增量训练 |