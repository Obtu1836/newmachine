import numpy as np
from numpy.typing import NDArray


from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline

'''
采用了 截距
隐藏层的激活函数 采用softplus函数 输出层采用sigmoid
'''

class Sigmoid:

    def forward(self, x: NDArray):
        x = np.clip(x, -250, 250)
        return 1/(1+np.exp(-x))

    def backward(self, x: NDArray):

        return self.forward(x)*(1-self.forward(x))


class Softplus:  # Softplus 的逻辑
    def forward(self, x: NDArray):
        return np.log1p(np.exp(np.clip(x, -250, 250)))

    def backward(self, x: NDArray):
        # Softplus 的导数是 Sigmoid
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))


class Model(ClassifierMixin, BaseEstimator):

    def __init__(self, max_iters: int = 10000,
                 lr: float = 1e-2,
                 hidden: int = 64):

        self.max_iters = max_iters
        self.lr = lr
        self.hidden = hidden

    def _loss(self):

        return -(self.y_*np.log(self.f2)+(1-self.y_)*np.log(1-self.f2)).sum()/self.m

    def fit(self, x: NDArray, y: NDArray):

        self.y_ = y[:, None]
        self.classes_, _ = np.unique(self.y_, return_counts=True)
        x = x[:, None] if x.ndim == 1 else x
        self.x = np.column_stack([x, np.ones(len(x))])# 对输入添加截距
        self.m, self.n = self.x.shape

        self.w1_ = np.random.randn(self.n, self.hidden)
        self.w2_ = np.random.randn(self.hidden + 1, 1) # 对隐藏层添加截距

        self.sigmoid = Sigmoid()
        self.softplus = Softplus()

        for i in range(self.max_iters):
            self.forward()
            self.backward()

        return self

    def forward(self):

        self.y1 = self.x.dot(self.w1_)  # (m,n) (n,k) -->(m,k)
        self.f1 = self.softplus.forward(self.y1)  # (m,k)
        self.f1_with_bias = np.column_stack(
            [self.f1, np.ones(self.m)])  # (m,k+1)

        self.y2 = self.f1_with_bias.dot(self.w2_)  # (m,k+1)(k+1,1)-->(m,1)
        self.f2 = self.sigmoid.forward(self.y2)

    def backward(self):
        '''
        关于err2中 残差项的来源 (self.f2-self.y) 

        交叉熵部分求导: (f2-y) /(f2*(1-f2))
        sigmoid部分 sigmoid(y2)*(1-sigmoid(y2))=f2*(1-f2)

        1 必须是 采用了交叉熵
        2 经过了激活函数 sigmoid

        这两个通过链式求导 分母正好抵消
        '''


        err2 = (self.f2-self.y_)/self.m  # (m,1)
        d_w2 = self.f1_with_bias.T.dot(err2)  # (k+1,m) (m,1) -->(k+1,1)

        # (m,1) (1,k) *(m,k)
        err1 = err2.dot(self.w2_[:-1, :].T)*self.softplus.backward(self.y1)
        d_w1 = self.x.T.dot(err1)  # (n,m) (m,k) -->(n,k)

        self.w2_ -= self.lr*d_w2
        self.w1_ -= self.lr*d_w1

    def predict(self, x: NDArray):

        x = x[:, None] if x.ndim == 1 else x
        x = np.column_stack([x, np.ones(len(x))])
        m = x.shape[0]
        y1 = x.dot(self.w1_)  # (m,n) (n,k) -->(m,k)
        f1 = self.softplus.forward(y1)  # (m,k)
        f1_with_bias = np.column_stack([f1, np.ones(m)])  # (m,k+1)

        yp = f1_with_bias.dot(self.w2_)

        prob = self.sigmoid.forward(yp).ravel()
        label = np.where(prob > 0.5, 1, 0)
        return label


def main():
    x, y = load_breast_cancer(return_X_y=True)
    x = np.asarray(x)
    y = np.asarray(y)

    sclaer = StandardScaler()
    clf = Model(max_iters=5000, lr=1e-2, hidden=128)
    model = Pipeline([('scaler', sclaer), ('clf', clf)])

    cv = StratifiedKFold(5, shuffle=True)
    results = cross_validate(model, x, y, cv=cv)

    print(results['test_score'])


if __name__ == '__main__':
    main()
