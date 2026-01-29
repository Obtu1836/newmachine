1. 常规参数 (General Parameters)
booster: 使用哪种弱学习器。常用 gbtree (树模型) 或 gblinear (线性模型)。
verbosity: 打印日志的详细程度。0 (silent), 1 (warning), 2 (info), 3 (debug)。
nthread: 并行线程数，通常不设置则默认使用最大可用核心。
device: 指定运行设备，如 cpu 或 cuda (GPU)。


2. 树提升器参数 (Tree Booster Parameters)
eta: (同 learning_rate) 学习率，范围 [0, 1]。防止过拟合的核心参数，通常设为 0.01-0.3。
max_depth: 树的最大深度，通常设为 3-10。
gamma: (同 min_split_loss) 节点分裂所需的最小损失减少量。越大模型越保守。
min_child_weight: 孩子节点中样本权重和的最小阈值。用于防止学习到局部噪声。
subsample: 训练每棵树时随机采样的样本比例，通常 0.5-1。
colsample_bytree: 构建树时随机采样的特征比例。
lambda: L2 正则化权重。默认值1 可设置（0-1000）
alpha: L1 正则化权重。默认值0 可设置 0-100
tree_method: 树构建算法。auto, exact, approx, hist (推荐大场景使用 hist)。

（优先调节 max_depth 和 learning_rate (eta)。如果搞不定过拟合，再把 lambda 往 10-100 范围内加点。）


3. 学习任务参数 (Learning Task Parameters)
objective: (必填) 目标函数：
    binary:logistic: 二分类，输出概率。
    multi:softmax: 多分类，输出类别标签（需设 num_class）。
    multi:softprob: 多分类，输出概率。
    reg:squarederror: 回归任务。
eval_metric: 评估指标（如 rmse, mae, logloss, error, auc, mlogloss）。
num_class: 如果是多分类，必须指定类别总数。
