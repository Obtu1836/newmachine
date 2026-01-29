import numpy as np 
from numpy.typing import NDArray
from typing import Literal
from pathlib import Path

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz

class Node:
    """
    决策树节点类
    """
    def __init__(self, column=None, value=None,
                 left=None, right=None, leaf=None):
        self.column = column    # 分裂特征的索引
        self.value = value      # 分裂特征的阈值
        self.left = left        # 左子树（小于阈值）
        self.right = right      # 右子树（大于等于阈值）
        self.leaf = leaf        # 如果是叶子节点，存储分类标签

    def is_leaf(self):
        """判断当前节点是否为叶子节点"""
        return self.leaf is not None
    
class DecisionTree:
    """
    自定义决策树分类器
    """
    def __init__(self, mode: Literal['Entropy', 'Gini'], max_depth: int,
                 min_samples_split: int = 5,
                 min_increase: float = 1e-2):
        self.mode = mode                            # 准则：'Entropy' (信息熵) 或 'Gini' (基尼系数)
        self.max_depth = max_depth                  # 树的最大深度（防止过拟合）
        self.min_samples_split = min_samples_split  # 节点分裂所需的最小样本数
        self.min_increase = min_increase            # 分裂所需的最小增益量（预剪枝）
        self.tree = None                            # 存储构建好的树根节点
    
    def _cal_metirc(self, label: NDArray):
        """计算指标（信息熵或基尼系数）"""
        if len(label) == 0: return 0
        _, nums = np.unique(label, return_counts=True)
        p = nums / nums.sum()

        if self.mode == 'Entropy':
            return sum(-p * np.log2(p))
        return 1 - np.power(p, 2).sum()
    
    def _split(self, data: NDArray, label: NDArray,
               column: int, value: float):
        """根据给定的列和阈值切分数据集"""
        mask = data[:, column] < value
        return data[mask], label[mask], data[~mask], label[~mask]
    
    def _best_split(self, data: NDArray, label: NDArray):
        """遍历所有特征和特征值，寻找最优分裂点"""
        increase = 0
        params = {}
        init_metric = self._cal_metirc(label)

        _, n = data.shape
        for col in range(n):
            features = np.unique(data[:, col])
            for val in features:
                l_data, l_lable, r_data, r_lable = self._split(data, label, col, val)
                if len(l_data) == 0 or len(r_data) == 0:
                    continue
                
                # 计算分裂后的加权指标
                l_metrc = self._cal_metirc(l_lable)
                r_metic = self._cal_metirc(r_lable)
                new_metric = (l_metrc * len(l_data) + r_metic * len(r_data)) / len(data)

                # 计算信息增益/基尼增益
                '''
                这一步 只要找到了划分 样本分裂以后的信息熵或者gini系数 一定比原来低
                因为这两个函数是概率分布函数 属于凹函数 jeston不等式 f(sum(x1,x2...))>=sum(fx1,fx2)
                '''
                diff = init_metric - new_metric  #所以 diff是>=0的
                # 记录最大增益的点
                if diff > increase and diff > self.min_increase:
                    increase = diff
                    params.update({'column': col, 'value': val,
                                   'l_data': l_data, 'r_data': r_data,
                                   'l_lable': l_lable, 'r_lable': r_lable})
            
        return params 
    
    def _calculate_leaf(self, label):
        """计算叶子节点的标签（取众数）"""
        lab, nums = np.unique(label, return_counts=True)
        tag = lab[np.argmax(nums)]
        return tag
    
    def build_tree(self, data: NDArray, label: NDArray, depth: int)->Node:
        """递归构建决策树"""
        # 停止条件：样本标签统一，或样本数少于阈值
        if len(np.unique(label)) == 1 or len(data) < self.min_samples_split:
            tag=self._calculate_leaf(label)
            return Node(leaf=tag)
        
        # 停止条件：达到最大深度
        if depth >= self.max_depth:
            tag=self._calculate_leaf(label)
            return Node(leaf=tag)
        
        # 寻找最优分裂参数
        params = self._best_split(data, label)

        # 停止条件：无法进一步分裂或增益太小
        if not params:
            tag=self._calculate_leaf(label)
            return Node(leaf=tag)
        
        # 递归构建左右子树
        left = self.build_tree(params['l_data'], params['l_lable'], depth + 1)
        right = self.build_tree(params['r_data'], params['r_lable'], depth + 1)
        
        return Node(column=params['column'], value=params['value'],
                    left=left, right=right)

    def fit(self, data: NDArray, label: NDArray):
        """训练入口"""
        self.tree = self.build_tree(data, label, 0)

    def _predict_one(self, tree: Node, test: NDArray):
        """对单个样本进行递归预测"""
        if tree.is_leaf():
            return tree.leaf
        
        assert tree.left is not None
        assert tree.right is not None
        if test[tree.column] < tree.value:
            return self._predict_one(tree.left, test)
        return self._predict_one(tree.right, test)
            
    def predict(self, tests):
        """批量预测"""
        assert self.tree is not None, '模型未训练'
        if len(tests)==0:
            return np.full(len(tests),0)
        res = np.array([self._predict_one(self.tree, test) for test in tests])
        return res
    
    def display(self):
        """以文本形式打印树结构"""
        def inner(tree: Node, level: str):
            if tree.is_leaf():
                print(level + str(tree.leaf))
            else:
                print(level + '*' + str(tree.column) + '*' + str(tree.value))
                assert tree.left is not None
                assert tree.right is not None
                inner(tree.left, level + 'L-')
                inner(tree.right, level + 'R-')

        assert self.tree is not None, '模型未训练'
        inner(self.tree, level='ROOT-')

    def visualize(self, filename='tree'):
        """使用 graphviz 生成树的可视化图片"""
        dot = graphviz.Digraph(comment='Decision Tree')

        def add_nodes(node: Node, parent_id=None, edge_label=""):
            node_id = str(id(node))
            if node.is_leaf():
                label = f"Leaf: {node.leaf}"
                dot.node(node_id, label=label, shape='ellipse', color='green')
            else:
                label = f"X[{node.column}] < {node.value:.2f}"
                dot.node(node_id, label=label, shape='box')

            if parent_id:
                dot.edge(parent_id, node_id, label=edge_label)

            if not node.is_leaf():
                assert node.left is not None
                assert node.right is not None
                add_nodes(node.left, node_id, "True")
                add_nodes(node.right, node_id, "False")

        assert self.tree is not None, '模型未训练'
        add_nodes(self.tree)
        dot.render(filename, format='png', cleanup=True)
        print(f"树结构已保存至 {filename}.png")

    def prune(self, x_val: NDArray, y_val: NDArray):
        """后剪枝入口"""
        if self.tree is None:
            return
        self.tree = self._prune_recursive(self.tree, x_val, y_val)

    def _prune_recursive(self, node: Node, x_val: NDArray, y_val: NDArray):
        # 如果验证集为空，停止剪枝
        if len(x_val) == 0:
            return node
        
        # 如果是叶子节点，无需剪枝
        if node.is_leaf():
            return node

        # 1. 递归向下剪枝（自底向上）
        mask = x_val[:, node.column] < node.value
        assert node.left is not None
        assert node.right is not None
        node.left = self._prune_recursive(node.left, x_val[mask], y_val[mask])
        node.right = self._prune_recursive(node.right, x_val[~mask], y_val[~mask])

        # 2. 评估当前子树的效果
        # 计算当前节点作为非叶子节点时在验证集上的性能
        y_pred_before = np.array([self._predict_one(node, x) for x in x_val])
        acc_before = accuracy_score(y_val, y_pred_before)

        # 3. 评估替换为叶子节点后的效果
        # 获取当前验证集样本在当前节点出现频率最高的类

        majority_label=self._calculate_leaf(y_val)
        # 计算假设为叶子节点时的准确率
        acc_after = accuracy_score(y_val, np.full(len(y_val), majority_label))

        # 4. 如果剪枝后准确率没有下降（或更高），则执行剪枝
        if acc_after >= acc_before:
            return Node(leaf=majority_label)
        
        return node

def main():
    # 加载数据集
    x, y = load_iris(return_X_y=True)
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, train_size=0.6)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, train_size=0.5)

    # 初始化并训练模型
    decs = DecisionTree('Entropy', 6, min_samples_split=2, min_increase=0)
    decs.fit(x_train, y_train)
    
    # 在测试前进行后剪枝
    decs.prune(x_val, y_val)

    yt = decs.predict(x_train)
    yp = decs.predict(x_test)

    # 打印评估指标
    print(f"训练集准确率: {accuracy_score(y_train, yt):.3f}")
    print(f"测试集准确率: {accuracy_score(y_test, yp):.3f}")

    # 可视化
    path = Path(r'classification/DecisionTree/iris_tree1')
    decs.visualize(str(path))

if __name__ == '__main__':
    main()












