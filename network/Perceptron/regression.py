import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Literal
from sklearn.model_selection import cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import RegressorMixin, BaseEstimator

import matplotlib.pyplot as plt


class Activate(ABC):
    @abstractmethod
    def forward(self, x) -> NDArray: ...
    @abstractmethod
    def backward(self, x) -> NDArray: ...


ALL_ACTIVATE: dict[str, type[Activate]] = {}


def auto_add(cls):
    name = str.lower(cls.__name__)
    ALL_ACTIVATE[name] = cls
    return cls


@auto_add
class Relu(Activate):

    def forward(self, x: NDArray):
        return np.maximum(0, x)

    def backward(self, x: NDArray):
        return (x > 0).astype(float)


@auto_add
class Sigmoid(Activate):

    def forward(self, x: NDArray):
        x = np.clip(x, -250, 250)
        return 1/(1+np.exp(-x))

    def backward(self, x: NDArray):

        return self.forward(x)*(1-self.forward(x))


@auto_add
class Tanh(Activate):
    def forward(self, x: NDArray):
        return np.tanh(x)

    def backward(self, x: NDArray):
        return 1.0 - np.tanh(x)**2


@auto_add
class Softplus(Activate):  # Softplus 的逻辑
    def forward(self, x: NDArray):
        return np.log1p(np.exp(np.clip(x, -250, 250)))

    def backward(self, x: NDArray):
        # Softplus 的导数是 Sigmoid
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))


class Model(RegressorMixin, BaseEstimator):

    def __init__(self, mode: Literal['relu', 'sigmoid', 'tanh', 'softplus'],
                 max_iters: int = 10000,
                 lr: float = 1e-2,
                 hidden: int = 64):

        self.mode = mode
        self.max_iters = max_iters
        self.lr = lr
        self.hidden = hidden

    def fit(self, x: NDArray, y: NDArray):

        self.scaler_ = StandardScaler()
        x = x[:, None] if x.ndim == 1 else x
        self.y_ = self.scaler_.fit_transform(y[:, None])
        self.m = len(self.y_)
        self.x = np.column_stack([x, np.ones(self.m)])
        self.m, self.n = self.x.shape

        # 使用 Xavier/He 初始化改进
        scale1 = np.sqrt(2.0 / self.n) if self.mode == 'relu' else np.sqrt(1.0 / self.n)
        scale2 = np.sqrt(1.0 / self.hidden)
        
        self.w1_ = np.random.randn(self.n, self.hidden) * scale1
        self.w2_ = np.random.randn(self.hidden + 1, 1) * scale2

        function = ALL_ACTIVATE.get(self.mode, Relu)
        self.fun = function()

        for i in range(self.max_iters):
            self.forward()
            self.backward()

        return self

    def forward(self):

        self.y1 = self.x.dot(self.w1_)  # (m,n) (n,k) -->(m,k)
        self.f1 = self.fun.forward(self.y1)  # (m,k)
        self.f1with_bias = np.column_stack(
            [self.f1, np.ones(self.m)])  # (m,k+1)

        self.y2 = self.f1with_bias.dot(self.w2_)  # (m,k+1)(k+1,1)-->(m,1)

    def backward(self):

        err2 = (self.y2-self.y_)/self.m  # (m,1)
        d_w2 = self.f1with_bias.T.dot(err2)  # (k+1,m) (m,1)  (k+1,1)

        err1 = err2.dot(self.w2_[:-1, :].T)*self.fun.backward(self.y1)
        d_w1 = self.x.T.dot(err1)

        self.w2_ -= self.lr*d_w2
        self.w1_ -= self.lr*d_w1

    def predict(self, x: NDArray):

        x = x[:, None] if x.ndim == 1 else x
        x = np.column_stack([x, np.ones(len(x))])
        m = x.shape[0]
        y1 = x.dot(self.w1_)  # (m,n) (n,k) -->(m,k)
        f1 = self.fun.forward(y1)  # (m,k)
        f1_with_bias = np.column_stack([f1, np.ones(m)])  # (m,k+1)

        yp = f1_with_bias.dot(self.w2_)
        yp = self.scaler_.inverse_transform(yp)

        return yp.ravel()


def main():
    nums = 80
    x = np.linspace(-5, 5, nums)
    y = np.sin(x)+np.random.randn(nums)*0.1

    testx = np.linspace(-5, 5, 200)

    scaler = StandardScaler()
    clf = Model('tanh', lr=0.01, hidden=128, max_iters=3000)
    model = Pipeline([('scaler', scaler), ('clf', clf)])

    cv_style = KFold(5, shuffle=True)
    resuluts = cross_validate(model, x[:, None], y, cv=cv_style)
    print(f"r2_score: {resuluts['test_score'].mean():.4f}")

    model.fit(x[:, None], y)

    yp = model.predict(testx[:, None])

    plt.scatter(x, y)
    plt.plot(testx, yp)
    plt.show()


if __name__ == '__main__':
    main()
