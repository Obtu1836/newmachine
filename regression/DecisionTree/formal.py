import sys
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from sklearn.datasets import make_regression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_score


@dataclass
class Node:
    column: int | None = None
    value: int | None = None
    left: "Node|None" = None
    right: "Node|None" = None
    leaf: "float|None" = None


class Cart(BaseEstimator, RegressorMixin):  # 继承这两个类(使用k折交叉验证)
    def __init__(self, min_samples_split: int = 2,
                 quan_size: int = 10,
                 max_depth: int = sys.maxsize,
                 min_samples_leaf:int=5):
        # 参数必须直接赋值给同名属性，以便 get_params 自动获取 （交叉验证要求）
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.quan_size = quan_size
        self.min_samples_leaf=min_samples_leaf

    def _split(self, x: NDArray, y: NDArray, col: int, val: float | int):

        mask = x[:, col] < val
        return x[mask], y[mask], x[~mask], y[~mask]

    def _best_split_params(self, x: NDArray, y: NDArray):

        increase = 0
        params = {}

        init_var = np.var(y)
        _, n = x.shape
        for col in range(n):
            features = np.unique(x[:, col])

            if len(features) > self.quan_size:
                quan_features = np.quantile(features, np.linspace(0.1, 0.9, self.quan_size))
            else:
                quan_features = features

            for val in quan_features:
                l_x, l_y, r_x, r_y = self._split(x, y, col, val)
                if len(l_y) < self.min_samples_leaf or len(r_y) < self.min_samples_leaf:
                    continue
                l_var = l_y.var()
                r_var = r_y.var()
                new_var = (l_var * len(l_y) + r_var * len(r_y)) / len(y)
                diff = init_var - new_var
                if diff > increase:
                    increase = diff
                    params = {'l_x': l_x, 'r_x': r_x,
                              'l_y': l_y, 'r_y': r_y,
                              'column': col, 'value': val}

        return params

    def _build_tree(self, x: NDArray, y: NDArray, depth: int):

        if len(x) <= self.min_samples_split:
            return Node(leaf=y.mean())

        if depth <= 0:
            return Node(leaf=y.mean())

        params = self._best_split_params(x, y)

        if not params:
            return Node(leaf=y.mean())

        left = self._build_tree(params['l_x'], params['l_y'], depth - 1)
        right = self._build_tree(params['r_x'], params['r_y'], depth - 1)

        return Node(params['column'], params['value'],
                    left=left, right=right)

    def fit(self, X, y):  # 使用大写 X 以符合标准 API （交叉验证）
        self.tree = self._build_tree(X, y, self.max_depth)
        return self

    def _predict(self, tree: Node, test: NDArray):
        if tree.leaf is not None:
            return tree.leaf
        if test[tree.column] < tree.value:
            return self._predict(tree.left, test)  # type: ignore
        return self._predict(tree.right, test)  # type: ignore

    def predict(self, tests): #交叉验证要求必须有predict/score方法

        results = [self._predict(self.tree, test) for test in tests]
        return np.array(results)


def main():
    x, y = make_regression(3500, 4, random_state=0)[:2]

    # 实例化你的模型
    cart = Cart(min_samples_split=5, quan_size=10, max_depth=6,
                min_samples_leaf=5)

    # 使用 cross_val_score 进行 5 折交叉验证
    # scoring='r2' 表示使用 R2 分数
    scores = cross_val_score(cart, x, y, cv=5, scoring='r2')

    print(f"每折 R2 分数: {scores}")
    print(f"平均 R2 分数: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")


if __name__ == '__main__':
    main()

