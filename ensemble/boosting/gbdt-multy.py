import numpy as np
from numpy.typing import NDArray

from ensemble.boosting.cart import Cart
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_score, accuracy_score, recall_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import StratifiedKFold


class LabelEncoder:
    '''
    LabelEncoder 的 Docstring
    将整数 不连续的标签 转为 从0开始的连续标签
    '''

    def fit(self, tags):

        self.lab, _ = np.unique(tags, return_counts=True)
        idx = np.arange(len(self.lab))
        self.mask = np.zeros(max(self.lab)+1, dtype=int)
        self.mask[self.lab] = idx

    def transform(self, tags):
        yp = self.mask[tags]
        return yp

    def fit_transform(self, tags):
        self.fit(tags)
        coderlabel = self.transform(tags)
        return coderlabel

    def inverse_transform(self, coderlabel): # 还原
        return self.lab[coderlabel]


class GBDT(RegressorMixin, BaseEstimator):

    def __init__(self, min_samples_leaf: int, min_samples_split: int,
                 max_depth: int, lr: float, n_trees: int):

        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.lr = lr
        self.n_trees = n_trees

        self.coder = LabelEncoder()
        self.trees = []

    def _softmax(self, x: NDArray):
        exp = np.exp(x-x.max(axis=1, keepdims=True)) #分子分母 同除最大值 值不变 有效防数值溢出
        return exp/np.sum(exp, axis=1, keepdims=True)

    def fit(self, x, y):
        code_label = self.coder.fit_transform(y) # 将标签转为从0开始连续的整数标签
        self.classes_, nums = np.unique(code_label, return_counts=True)
        self.init_value = np.log(nums/nums.sum()+1e-6)#初始值为按样本各个类别的样本数量比,形状（k,)
        m, _ = x.shape
        preds = np.tile(self.init_value, (m, 1)) # 将初始值 推广为(m,k)
        k_nums = len(self.classes_)
        binary_ylabel = np.eye(k_nums)[code_label] # 二值化 标签

        for i in range(self.n_trees):
            softmax_pred = self._softmax(preds) # 将预测过一下softmax
            k_trees = []
            outputs = np.zeros((m, k_nums)) #临时保存类别树的预测结果 方便后续的整体更新
            for k in range(k_nums):
                k_grad = softmax_pred[:, k]-binary_ylabel[:, k]
                tree = Cart(self.min_samples_leaf, self.max_depth,
                            self.min_samples_split, 7)
                tree.fit(x, k_grad)
                output = tree.predict(x)
                outputs[:, k] = output   #保存
                k_trees.append(tree)
            preds -= self.lr*outputs # 当所有类别都预测结束以后更新
            self.trees.append(k_trees)

        return self

    def predict_proba(self, test):

        pred = np.tile(self.init_value, (len(test), 1))
        for ktrees in self.trees:
            for i, tree in enumerate(ktrees):
                pred[:, i] -= self.lr*tree.predict(test)

        softmax_pred = self._softmax(pred)
        return softmax_pred

    def predict(self, test):
        
        proba = self.predict_proba(test)
        binary_label = proba.argmax(axis=1)
        label = self.coder.inverse_transform(binary_label)
        return label


def main():
    x, y = load_wine(return_X_y=True)
    x = np.asarray(x)

    model = GBDT(5, 7, 5, 0.1, 100)

    stkfold = StratifiedKFold(5, shuffle=True)
    precision = make_scorer(precision_score, average='macro')
    recall = make_scorer(recall_score, average='macro')
    score = {'precision': precision,
             'acc': 'accuracy',
             'recall': recall}
    results = cross_validate(model, x, y, cv=stkfold, scoring=score)

    print(f"{'macro_precision':<20} {results['test_precision'].mean():.3f}")
    print(f"{'macro_recall':<20} {results['test_recall'].mean():.3f}")
    print(f"{'acc':<20} {results['test_acc'].mean():.3f}")


if __name__ == '__main__':
    main()
