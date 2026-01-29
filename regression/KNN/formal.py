import numpy as np
from abc import ABC,abstractmethod
from numpy.typing import NDArray
from numpy.linalg import solve, norm
from typing import Literal
from sklearn.datasets import load_diabetes
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


class Strategy(ABC):
    
    def __init__(self,x:NDArray,y:NDArray):

        self.x=x
        self.y=y

    @abstractmethod
    def predict(self,*args,**kwargs):...

ALL_STRATEGY :dict[str,type[Strategy]]= {}

def autoaddstrategy(fun):
    name = str.lower(fun.__name__)
    ALL_STRATEGY[name] = fun
    return fun


@autoaddstrategy
class Mean(Strategy):
 
    def predict(self, test: NDArray, k: int, **kwargs):
        test =  np.c_[test, np.ones(len(test))]
        dis = norm(test[:, None]-self.x, axis=2)  # (m,n)
        min_dis_idx = np.argpartition(dis, k, 1)[:, :k]
        ytag = self.y.ravel()[min_dis_idx]
        return ytag.mean(axis=1)


@autoaddstrategy
class Linear(Strategy):
   
    def predict(self, test: NDArray, k: int, **kwargs):
        test = np.c_[test, np.ones(len(test))]

        dis = norm(test[:, None]-self.x, axis=2)
        min_dis_idx = np.argpartition(dis, k, 1)[:, :k]  # (m,k)
        bx = self.x[min_dis_idx]  # (m,k,r)
        by = self.y[min_dis_idx]  # (m,k,1)
        bxt = np.transpose(bx, (0, 2, 1))  # (m,r,k)
        bxtx = bxt@bx  # (m,r,r)
        reg = np.eye(bxtx.shape[-1])*1e-3
        bw = solve(bxtx+reg, bxt@by)  # (m,r,1)

        yp = test[:, None, :]@bw
        return yp.ravel()


@autoaddstrategy
class Lwlr(Strategy):

    def predict(self, test: NDArray, k: int,sigma: float = 0.2,**kwargs):
        test = np.c_[test, np.ones(len(test))]
        dis = np.sum(np.power(test[:, None]-self.x, 2), axis=2)
        min_dis_idx = np.argpartition(dis, k, 1)[:, :k]  # (m,k)
        idx = np.arange(len(test))[:, None]
        min_dis = dis[idx, min_dis_idx] #(m,k)
        min_dis = np.exp(-min_dis/(2*sigma**2))  # (m,k)

        bx = self.x[min_dis_idx]  # (m,k,r)
        bxt = np.transpose(bx, (0, 2, 1))  # (m,r,k)
        bxtq = bxt*min_dis[:, None, :]  # (m,r,k)
        bxtqx = bxtq@bx  # (m,r,r)
        reg = np.eye(bxtqx.shape[-1])*1e-2
        by = self.y[min_dis_idx]  # (m,k,1)
        bw = solve(bxtqx+reg, bxtq@by)  # (m,r,1)

        yp = test[:, None, :]@bw
        return yp.ravel()


class Knn(BaseEstimator, RegressorMixin): # 继承这两个类

    def __init__(self, k: int=5, strategy: Literal['mean', 'linear', 'lwlr']='lwlr',
                 sigma:float=0.1):
        # 参数必须直接赋值给同名属性
        self.k = k
        self.strategy = strategy
        self.sigma = sigma

    def fit(self, X: NDArray, y: NDArray): # 使用标准 X, y 命名
        
        self.trainx = np.c_[X, np.ones(len(X))]
        self.trainy = y[:, None] if y.ndim == 1 else y
        
        instance_classx= ALL_STRATEGY.get(self.strategy) #必须创建一个实例
        if instance_classx is None:
            raise RuntimeError('strategy not exits')
        self.model_=instance_classx(self.trainx,self.trainy) #初始化实例
        return self # 必须返回 self

    def predict(self, X: NDArray):
        # i· RuntimeError('model must be fitted')
        # 内部调用策略的 predict
        yp = self.model_.predict(X, self.k, sigma=self.sigma)
        return yp

def main():
    x, y = load_diabetes(return_X_y=True)
    x=np.asarray(x)
    
   
    scaler=MinMaxScaler()
    clf=Knn(k=50,strategy='lwlr',sigma=0.5)

    model=Pipeline([('scaler',scaler),('clf',clf)])
   
    scores=cross_validate(model,x,y,cv=5,scoring='r2')

    print(f"每折 R2 分数: {scores['test_score'].mean()}")

if __name__ == '__main__':
    main()
