import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax
from sklearn.datasets import load_iris, load_digits
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, StratifiedKFold

class Softplus:  # Softplus 的逻辑
    def forward(self, x: NDArray):
        return np.log1p(np.exp(np.clip(x, -250, 250)))

    def backward(self, x: NDArray):
        # Softplus 的导数是 Sigmoid
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))


class Model(ClassifierMixin, BaseEstimator):
    def __init__(self, max_iters: int = 5000, hidden: int = 128, lr=1e-2):

        self.max_iters = max_iters
        self.hidden = hidden
        self.lr = lr

    def fit(self, x: NDArray, y: NDArray):

        x = x[:, None] if x.ndim == 1 else x
        self.y = y[:, None] if y.ndim == 1 else y

        self.classes_, _ = np.unique(y, return_counts=True)
        self.k_class = len(self.classes_)
        self.mask = np.eye(self.k_class)[y]
        self.x = np.column_stack([x, np.ones(len(x))])
        self.m, self.n = self.x.shape

        self.w1_ = np.random.randn(self.n, self.hidden)
        self.w2_ = np.random.randn(self.hidden+1, self.k_class)

        self.softplus = Softplus()

        for i in range(self.max_iters):
            self.forward()
            self.backward()

    def forward(self):

        self.y1 = self.x.dot(self.w1_)  # (m,h)
        self.f1 = self.softplus.forward(self.y1)  # (m,h)

        self.f1_with_bias = np.column_stack(
            [self.f1, np.ones(self.m)])  # (m,h+1)
        self.y2 = self.f1_with_bias.dot(self.w2_)  # (m,h+1) (h+1,k)-->(m,k)

        self.f2 = softmax(self.y2, axis=1)  # (m,k)

    def backward(self):
        '''
        err2 残差项的来源 
        softmax和交叉熵共同作用 证明 可参考实现逻辑回归的softmax的求导
        '''
        err2 = (self.f2-self.mask)/self.m  # (m,k)
        g_w2 = self.f1_with_bias.T.dot(err2)  # (h+1,m)(m,k) -->(h+1,k)

        # (m,k) (k,h)  * (m,h)
        err1 = err2.dot(self.w2_[:-1, :].T)*self.softplus.backward(self.y1)
        g_w1 = self.x.T.dot(err1)

        self.w2_ -= self.lr*g_w2
        self.w1_ -= self.lr*g_w1

        return self

    def predict_proba(self, test: NDArray):

        test = test[:, None] if test.ndim == 1 else test
        m = len(test)
        test = np.column_stack([test, np.ones(m)])
        y1 = test.dot(self.w1_)
        f1 = self.softplus.forward(y1)
        f1_bias = np.column_stack([f1, np.ones(m)])
        y2 = f1_bias.dot(self.w2_)

        proba = softmax(y2, axis=1)

        return proba

    def predict(self, test):

        proba = self.predict_proba(test)
        label = proba.argmax(axis=1)

        return self.classes_[label]


def main():

    x,y=load_iris(return_X_y=True)
    # x, y = load_digits(return_X_y=True)
    x, y = np.asarray(x), np.asarray(y)

    scaler = StandardScaler()
    clf = Model(1000)

    model = Pipeline([('scaler', scaler), ('clf', clf)])

    model.fit(x, y)

    cv_style = StratifiedKFold(5, shuffle=True)

    results = cross_validate(model, x, y, cv=cv_style)

    print(results['test_score'])


if __name__ == '__main__':
    main()
