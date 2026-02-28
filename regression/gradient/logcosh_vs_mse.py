import numpy as np
from pathlib import Path
from numpy.typing import NDArray
from abc import ABC, abstractmethod

from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# 双曲余弦对数公式：np.sum(np.log((np.exp(yp-y)+np.exp(-(yp-y)))/2)) 

'''
收敛速度对比：
MSE 初始下降剧烈 因为初始训练时时误差较大 梯度也就大 下降剧烈
logcosh 因为始终限制在[-1,1] 在误差较大时比较恒定 避免了梯度爆炸 所以平稳性强 最终需要的迭代次数也相对多
'''


class BaseRegression(ABC):
    def __init__(self, lr: float = 1e-3, max_iters: int = 2000):

        self.lr = lr
        self.max_iters = max_iters
        

    @abstractmethod
    def _cal_loss(self, error: NDArray) -> float:
        pass

    @abstractmethod
    def _cal_grad(self, error: NDArray) -> NDArray:
        pass

    def fit(self, x, y):

        x = np.atleast_2d(x)
        m, _ = x.shape
        self.x = np.column_stack([x, np.ones(m)])
        m, n = self.x.shape

        self.w = np.zeros((n, 1))

        self.y = y[:, None]

        self.lossCollection = []

        for i in range(self.max_iters):
            error = self.x.dot(self.w)-self.y
            self.w -= (1/m)*self.lr*self._cal_grad(error)
            self.lossCollection.append((1/m)*self._cal_loss(error))

    def predict(self, test):
        test = np.atleast_2d(test)
        test = np.column_stack([test, np.ones(len(test))])

        return (test.dot(self.w)).ravel()


class MseRegerssor(BaseRegression):

    def _cal_loss(self, error: NDArray) -> float:
        return error.T.dot(error)

    def _cal_grad(self, error: NDArray) -> NDArray:
        return self.x.T.dot(error)
    


class LogCoshRegressor(BaseRegression):

    def _cal_loss(self, error:NDArray) -> float:

        #第一种写法（按照定义）：
        # loss=np.log(np.cosh(error))
        #数值稳定版本
        loss=np.abs(error) + np.log1p(np.exp(-2 * np.abs(error))) - np.log(2)
        return np.sum(loss)
        
    def _cal_grad(self, error):
        # 直接使用 tanh 即可，它天生稳定，且这就是上述稳定版公式的数学导数
        # 无论采取上述哪种损失函数写法 导函数一致
        return self.x.T.dot(np.tanh(error))
    
class Paint:
    def __init__(self):

        fig,self.ax=plt.subplots()
        path=Path(__file__).resolve().parent
        self.save_path=path/'mse_logcosh.png'

        self.ax.set_yscale('log')

        
    def drawing(self,loss_mse,loss_logcosh):

        self.ax.plot(np.arange(len(loss_mse)),loss_mse,label='mse')
        self.ax.plot(np.arange(len(loss_logcosh)),loss_logcosh,label='logcosh')
        self.ax.legend()

        plt.savefig(self.save_path)
        plt.show()

def main():
    x,y=make_regression(2400,2,noise=10)[:2]

    scaler=StandardScaler()
    x=scaler.fit_transform(x)

    mse_clf=MseRegerssor(lr=0.1)
    mse_clf.fit(x,y)
    loss1=np.array(mse_clf.lossCollection).ravel()

    log_cosh_clf=LogCoshRegressor(lr=0.1)
    log_cosh_clf.fit(x,y)
    loss2=np.array(log_cosh_clf.lossCollection).ravel()

    paint=Paint()
    paint.drawing(loss1,loss2)
    

if __name__ == '__main__':
    main()


    

