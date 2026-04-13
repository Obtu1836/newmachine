# NewMachine

一个以机器学习算法原理为核心的实验型代码仓库，主要内容是常见监督学习、无监督学习、集成学习、概率模型和基础优化方法的手写实现。仓库更适合作为学习、推导、对照实验和小规模验证的资料集合，而不是生产环境中的通用机器学习库。

## 项目定位

- 目标是理解算法原理，而不是封装稳定 API。
- 多数脚本都可以单独运行，用于验证公式、观察训练过程或绘图分析。
- 部分实现参考了 sklearn 的接口风格，便于放入 Pipeline 或交叉验证流程中。
- 仓库中同时包含纯数值推导脚本、可视化脚本、实验性实现和少量说明文档。

## 主要内容

### 分类 classification

- Bayes
	- 高斯朴素贝叶斯
	- 多项式朴素贝叶斯
- DecisionTree
	- 决策树构建、剪枝、可视化
- KNN
	- K 近邻分类
- Label_Encoder
	- 标签编码相关实现
- Logic
	- sigmoid、softmax、log loss
	- 逻辑回归及优化器实现

### 聚类 cluster

- agnes
	- 凝聚层次聚类的不同实现版本
- distance
	- L0、L1、L2、余弦距离
- evaluation
	- 聚类标签匹配等评估方法
- kmeans
	- KMeans、KMedoids、TwoMeans、轮盘初始化、动画展示

### 集成学习 ensemble

- bagging
- boosting
	- CART、GBDT 二分类、多分类、回归
- voting
- xgb
	- XGBoost 参数和示例

### 评估指标 metric

- classification
	- F1、ROC、PR、One-vs-One、One-vs-Rest
- multilabel_classification
	- Jaccard、多标签评估示例

### 神经网络与优化 network

- Perceptron
	- 感知器二分类、回归、softmax
- optimize
	- 自适应步长、二维/三维优化过程展示
- mulgauss
	- 多高斯相关实验

### 概率模型 probamodel

- Em
	- EM 算法及向量化实现
- GaussMix
	- 高斯混合模型的分类与聚类示例

### 回归 regression

- gradient
	- 梯度下降回归
- leastsq
	- 最小二乘法
- lwlr
	- 局部加权线性回归
- newton
	- 牛顿法与泰勒展开相关实验
- poly
	- 多项式回归
- DecisionTree
	- 回归树
- KNN
	- K 近邻回归
- 解方程
	- 迭代法、牛顿法、梯度下降法

## 仓库结构

```text
newmachine/
├── classification/    分类算法
├── cluster/           聚类算法
├── ensemble/          集成学习
├── metric/            评估指标
├── network/           神经网络与优化
├── probamodel/        概率模型
├── regression/        回归与数值优化
├── test.ipynb         交互式实验 Notebook
├── test1.py           自适应步长可视化示例
├── pyproject.toml     项目依赖定义
└── commit.md          最近一次修复记录
```

## 环境要求

- Python 3.12 及以上
- 建议使用 uv 管理环境
- 也可以使用 venv + pip

当前 pyproject.toml 中声明的主要依赖包括：

- numpy
- pandas
- matplotlib
- scikit-learn
- sympy
- torch
- xgboost
- graphviz
- jax

注意：并不是所有脚本都会同时依赖这些库，但为了避免运行时缺包，建议先完整安装。

## 安装方式

### 使用 uv

```bash
uv sync
source .venv/bin/activate
```

### 使用 pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas matplotlib scikit-learn sympy torch xgboost graphviz jax
```

## 运行方式说明

这个仓库里的脚本有两种常见运行方式。

### 1. 直接运行文件

适用于没有包内相对导入的脚本，例如仓库根目录下的独立实验文件：

```bash
python test1.py
```

### 2. 以模块方式运行

如果某个文件使用了包内相对导入，例如：

```python
from .optimizer import create_optimizer
```

那么这类文件通常不能直接用 `python 文件路径.py` 运行，否则容易出现导入错误。应当在仓库根目录下使用模块方式执行：

```bash
python -m 包路径.模块名
```

例如：

```bash
python -m classification.Logic.optimize._logic
python -m regression.gradient.formal
```

建议先进入仓库根目录再执行命令：

```bash
cd /Users/mac/Code/newmachine
source .venv/bin/activate
python -m classification.Logic.optimize._logic
```

简单判断规则：

- 文件里如果出现 `from .xxx import ...` 或 `from ..xxx import ...`，优先使用模块方式运行。
- 位于仓库根目录、且不依赖相对导入的独立脚本，通常可以直接运行。

## 运行示例

仓库中的大部分脚本都带有 main 入口。下面列出几个代表性示例，并区分推荐的运行方式。

### 1. 逻辑回归与优化器实验

```bash
python -m classification.Logic.optimize._logic
```

这个脚本会：

- 读取 iris 数据集
- 使用 MinMaxScaler 做预处理
- 训练自定义逻辑回归分类器
- 通过交叉验证输出测试分数

### 2. 自适应步长可视化

```bash
python test1.py
```

这个脚本会绘制：

- 原函数曲线
- 梯度大小变化
- 自适应步长变化趋势

如果在 macOS 上运行时出现 Matplotlib 图形窗口阻塞，通常与 plt.show() 的交互式显示有关。更稳妥的做法是改为保存图片，或在图窗弹出后正常关闭窗口，不要直接中断进程。

### 3. 梯度下降回归

```bash
python -m regression.gradient.formal
```

### 4. 其他脚本

可以直接进入对应目录，运行其中带有 main 入口的文件，例如：

- 决策树相关 formal.py
- GBDT 相关脚本
- EM 与高斯混合模型脚本
- 感知器和 softmax 脚本

如果这些文件内部有相对导入，请统一从仓库根目录使用 `python -m ...` 的方式执行。

## 适合的使用方式

- 对照教材或公式推导阅读源码
- 修改超参数后直接运行脚本观察结果
- 用小数据集验证优化器、损失函数和梯度公式
- 把部分 sklearn 风格实现接入 Pipeline 做实验

## 当前代码风格特点

- 以脚本为单位组织内容，主题清晰，便于逐个试验
- 中文注释较多，更偏向学习笔记风格
- 不同目录下的代码成熟度不同，有的偏完整实现，有的偏局部验证
- 命名和接口风格并不完全统一，这是实验仓库常见现象

## 已知限制

- 这是实验仓库，不是生产级机器学习框架
- 没有系统化测试体系，正确性主要依赖示例运行和人工验证
- 没有统一的数据加载、训练入口和配置系统
- 部分代码更适合小规模数据，未针对大规模场景做性能优化
- 个别脚本包含绘图逻辑，在无图形界面的环境下需要改为保存图片
- 某些实现仍在迭代中，接口和行为可能继续调整

## 最近一次已记录修改

commit.md 中记录的最近修改为：修复逻辑回归中带权重参数的损失计算方式，以及梯度更新的计算方式。

## 建议阅读顺序

如果是第一次阅读这个仓库，建议按下面顺序浏览：

1. regression 和 classification 中的基础模型
2. network/optimize 中的优化可视化脚本
3. cluster 中的距离度量与 KMeans
4. ensemble 中的 GBDT 与投票方法
5. probamodel 中的 EM 和高斯混合模型

## 说明

这个仓库更像一份持续积累的算法实验笔记。若你的目标是学习原理、核对实现细节、快速做局部实验，这里的内容是合适的；若你的目标是直接复用成熟组件，建议优先使用 sklearn、xgboost、pytorch 等成熟库。
