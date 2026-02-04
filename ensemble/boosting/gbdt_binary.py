import numpy as np

from ensemble.boosting.cart import Cart
from sklearn.datasets import load_breast_cancer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score

class Encoder:
    def fit(self, tags):
        self.uni = np.unique(tags)
        idx = np.arange(len(self.uni))
        self.mask = np.zeros(max(self.uni)+1, dtype=int)
        self.mask[self.uni] = idx

    def transform(self, tags):
        yp = self.mask[tags]
        return yp

    def fit_tansform(self, tags):
        self.fit(tags)
        return self.transform(tags)

    def inverse_transform(self, label):

        return self.uni[label]


class GBDT_Binary(ClassifierMixin, BaseEstimator):

    def __init__(self, n_trees: int,
                       min_samples_split: int,
                       min_samples_leaf: int, 
                       max_depth: int,
                       lr: float,
                       quan_size:int=10,
                       weights:bool=False):

        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.lr = lr
        self.weights=weights

        self.quan_size=quan_size

        self.encoder = Encoder()

    def _sigmoid(self, x):

        x = np.clip(x, -250, 250)# 截断 防止数值溢出
        return 1/(1+np.exp(-x))

    def fit(self, x, y):

        encode_label = self.encoder.fit_tansform(y)

        self.classes_, nums = np.unique(encode_label, return_counts=True)
        p = (nums/nums.sum())

        posp=p[1] # 正样本的概率

        '''
        init_value是将原始概率经 (sigmoid反函数)的值 因为 在计算下面grad时 需要sigmoid(init_value)
        也就是通过先反函数再正 让模型始于一个较好的初值 也就是按照样本的分布 

        '''
        self.init_value = np.log(posp/(1-posp))
        pred = np.full(len(y), fill_value=self.init_value)#初始值 全部填充 正样本的概率值

        if self.weights: 
            weight=1/(p[encode_label]) #设置权重  与样本数量反比关系

        self.trees: list[Cart] = []

        for _ in range(self.n_trees):
            
            '''
            因为 初始值 填的的是sigmoid的反函数的值 所以经过sigmoid以后 
            正好是 原始样本分布的概率值
            grad=sigmoid(pred)-y  标准二分类模型损失函数
            '''
            if self.weights:
                grad = (self._sigmoid(pred)-encode_label)*weight
            else:
                grad=self._sigmoid(pred)-encode_label  #
            '''对每一棵树 训练残差拟合 提高模型精度'''
            tree = Cart(quan_size=self.quan_size,
                        min_samples_leaf=self.min_samples_leaf,
                        min_samples_split=self.min_samples_split,
                        max_depth=self.max_depth)
            
            tree.fit(x, grad)
            output = tree.predict(x)
            pred -= self.lr*output  #梯度下降式 迭代更新

            self.trees.append(tree)

        return self

    def predict_proba(self, test):
        pred = np.full(len(test), self.init_value)
        for tree in self.trees:
            pred -= self.lr*tree.predict(test)

        ypred = self._sigmoid(pred)
        return ypred

    def predict(self, test):

        proba = self.predict_proba(test)
        yp = np.where(proba > 0.5, 1, 0)

        label = self.encoder.inverse_transform(yp)
        return label


def main():

    x, y = load_breast_cancer(return_X_y=True)
    x = np.asarray(x)

    model = GBDT_Binary(50, 5, 3, 5, 0.1,8,True)
    cv_shuff = StratifiedKFold(n_splits=5, shuffle=True,random_state=0)
    score = make_scorer(recall_score, pos_label=0)

    scorin = {'acc': 'accuracy',
              'recall_pos': 'recall',
              'recall_neg': score}

    results = cross_validate(model, x, y, cv=cv_shuff, scoring=scorin)

    print(f"{'recall_neg:':<20}{results['test_recall_neg'].mean():.3f}")
    print(f"{'recall_pos:':<20}{results['test_recall_pos'].mean():.3f}")
    print(f"{'acc:':<20}{results['test_acc'].mean():.3f}")


if __name__ == '__main__':
    main()
