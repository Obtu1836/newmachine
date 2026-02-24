import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.metrics import make_scorer, jaccard_score
from sklearn.model_selection import cross_validate, KFold
from sklearn.datasets import make_multilabel_classification


'''
多标签分类任务 更具体的说是 二元关联方法

基分类器为 一个带正则项的二元逻辑回归 使用sigmoid和梯度下降
训练权重

核心策略是 为每个类别(标签)分别训练一个二元分类器 判断样本是否包含该标签
one-vs-rest

'''

class MultiLabelClassifer:
    '''
    二元逻辑回归
    '''
    def __init__(self, max_iters: int, lr: float, reg: float):

        self.max_iters = max_iters
        self.lr = lr
        self.reg = reg

    def _sigmoid(self, x):
        cx = np.clip(x, -250, 250)
        return 1/(1+np.exp(-cx))

    def fit(self, x, y):

        x = np.column_stack([x, np.ones(len(x))])#增加截距项
        self.m, self.n = x.shape
        self.w = np.zeros((self.n, 1))

        y = y[:, None]

        for i in range(self.max_iters):
            '''
            计算梯度 截距项不参与正则化
            '''
            grad = (1/self.m)*(x.T.dot(self._sigmoid(x.dot(self.w))-y))
            grad[:-1] += self.reg*self.w[:-1]

            self.w -= self.lr*grad

    def predict_proba(self, test):

        m = len(test)
        test = np.column_stack([test, np.ones(m)])
        yp = self._sigmoid(test.dot(self.w))
        return yp.ravel()


class Model(ClassifierMixin, BaseEstimator):

    def __init__(self, max_iters: int, lr: float, reg: float = 1e-2,
                 use_chain:bool=False):

        self.max_iters = max_iters
        self.lr = lr
        self.reg = reg
        self.use_chain=use_chain# 这个参数 决定是否使用链式
        '''使用链式 需要考虑标签的顺序问题 可根据随机或者相关性矩阵重新排序 
           当前代码并没有使用上述方法 只是按照原有的默认顺序'''

    def fit(self, x, y):

        _, n = y.shape
        self.classes_ =[np.unique(y[:,i]) for i in range(n)]
        self.k = len(self.classes_)
        self.dicts: dict[int, MultiLabelClassifer] = {}
        '''对每个类别 分别用基分类器 训练样本数据'''
        if not self.use_chain:
            for i in range(self.k):
                clf = MultiLabelClassifer(self.max_iters, self.lr, self.reg)
                clf.fit(x, y[:, i])
                self.dicts[i] = clf
        else:
            # 如果使用链式 要把每次样本标签 拼接到样本数据
            current_x=x.copy()
            for i in range(self.k):
                clf = MultiLabelClassifer(self.max_iters, self.lr, self.reg)
                clf.fit(current_x, y[:, i])
                self.dicts[i] = clf
                current_x=np.column_stack([current_x,y[:,i]])

        return self

    def predict_proba(self, test):
        m, _ = test.shape
        proba = np.zeros((m, self.k))

        if not self.use_chain:
            for idx, clf in self.dicts.items():
                proba[:,idx] = clf.predict_proba(test)

            return proba
        else:
            # 同理 预测时  也需要把每次预测值转成离散型后 继续添加到测试集
            currect_test=test.copy()
            for idx, clf in self.dicts.items():
                yp = clf.predict_proba(currect_test)
                proba[:,idx]=yp
                currect_test=np.column_stack([currect_test,(yp>0.5).astype(int)])#离散化
            return proba
            

    def predict(self, test):

        proba = self.predict_proba(test)
        label = (proba > 0.5).astype(int)

        return label


def main():
    x, y, _, _ = make_multilabel_classification(1800, 12, n_classes=5, n_labels=2,
                                                return_distributions=True, allow_unlabeled=False,
                                                random_state=18)
    x = np.asarray(x)

    scaler = StandardScaler()
    clf = Model(2000, 1e-2, 1e-2,True)

    model = Pipeline([('scaler', scaler), ('clf', clf)])

    cv_style = KFold(5, shuffle=True,random_state=21)

    jacc = make_scorer(jaccard_score, average='samples')
    score = {'jacc': jacc}

    results = cross_validate(model, x, y, cv=cv_style, scoring=score)

    print(results['test_jacc'].mean())


if __name__ == '__main__':
    main()
