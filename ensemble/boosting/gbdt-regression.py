import numpy as np
from numpy.typing import NDArray
from ensemble.boosting.cart import Cart
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

'''
以 self.mse 方式 简单理解
初始化预测标签
不断通过树 拟合 残差（梯度)  通过树 训练数据时 标签为残差
通过梯度下降的方式 更新预测标签
'''


class GBDTregressor:
    def __init__(self, n_estimators: int, lr: float,
                 max_depth: int, min_samples: int):

        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples = min_samples

        self.trees: list[Cart] = []

    def fit(self, x: NDArray, y: NDArray):
        #设置初始化数值 使用self-mse的损失函数 初始化时常用 均值 sum(((yp-ytrue)**2)
        #             使用self-mae作为损失函数 初始化时常用 中位数 sum(abs(yp-ytrue))

        self.init_prediction = np.mean(y) 
        # self.init_prediction=np.median(y) # mae de 初始化策略
        currect_prediction = np.full(len(y), self.init_prediction)

        for _ in range(self.n_estimators):

            residuals = currect_prediction-y
            # residuals=np.sign(currect_prediction-y) # mae时的梯度
            tree = Cart(self.min_samples, self.max_depth)
            tree.fit(x, residuals)  # 使用 每棵树 拟合程度的梯度
            predictions = tree.predict(x)

            currect_prediction -= self.lr*predictions #梯度下降法 迭代更新
            self.trees.append(tree)

    def predict(self, testx: NDArray):

        y_pred = np.full(len(testx), self.init_prediction)
        for tree in self.trees:
            y_pred -= self.lr*tree.predict(testx)

        return y_pred


def main():

    x, y = make_regression(1200, 4)[:2]
    trainx, testx, trainy, testy = train_test_split(x, y, train_size=0.8)

    cart = Cart(10,5)
    cart.fit(trainx, trainy)
    cart_yp = cart.predict(testx)

    print(f"{'单棵树r2:':<8}: {r2_score(testy, cart_yp):.3f}")

    gbdt = GBDTregressor(50, 0.5, 5,5)
    gbdt.fit(trainx, trainy)
    gbdt_yp = gbdt.predict(testx)

    print(f"{"集成树r2:":<8}: {r2_score(testy, gbdt_yp):.3f}")


if __name__ == '__main__':
    main()
