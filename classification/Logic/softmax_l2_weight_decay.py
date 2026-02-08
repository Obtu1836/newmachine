import numpy as np 
from numpy.linalg import solve,norm
from scipy.special import softmax
from sklearn.datasets import load_wine,load_iris
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.model_selection import cross_validate,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

class Logic(BaseEstimator,ClassifierMixin):

    def __init__(self,decay:float=1e-3,reg:float=1e-5,
                 max_iters:int=50,lr:float=1):

        self.decay=decay # 权重衰减 系数
        self.reg=reg # 阻尼牛顿法 正则项 确保正定
        self.max_iters=max_iters #最大迭代次数 因为牛顿法收敛快 一般不需奥设置太大
        self.lr=lr # 一般就是1
    
    def _softmax(self,x,w):
        
        c=x.dot(w)
        return softmax(c,axis=1)
        
        
    def _loss(self, x, w, mask):
        # 计算原始交叉熵损失
        probs = self._softmax(x, w)
        # 加入 1e-12 这种微小值防止 log(0) 报错
        cross_entropy = -np.sum(mask * np.log(probs + 1e-12)) / self.m
        
        # 增加 L2 正则项: (lambda / 2) * sum(w^2)
        # 通常不惩罚偏置项，但如果把偏置合并在 w 里了，全写进去也行
        # 权重衰减就是 np.sum(np.power(w,2))
        l2_penalty = 0.5 * self.decay * np.sum(w**2)
        
        return cross_entropy + l2_penalty

    def first_grad(self, x, w):
        # 增加 L2 正则化的梯度: grad + decay * w
        grad = -x.T.dot(self.mask - self._softmax(x, w)) / self.m
        #正则项的导数 就是本身  计算导数的过程就是逐元素求导(开方)
        return grad + self.decay * w 
    
    def second_grad(self, x, w):
        # n, k = w.shape
        hessian = np.zeros((self.n * self.k_nums,self.n * self.k_nums))
        probs = self._softmax(x, w)
        for i in range(self.m):
            xi_xt = np.outer(x[i], x[i])
            pi = probs[i]
            s_i = np.diag(pi) - np.outer(pi, pi)
            hessian += np.kron(xi_xt, s_i)
        
        #  Hessian 也要加上 L2 正则项的二阶导
        return (hessian / self.m) + (self.decay+self.reg) * np.eye(self.n * self.k_nums)

    def fit(self, x, y):

        self.classes_ = np.unique(y)
        self.k_nums = len(self.classes_)
        #  增加偏置项（Intercept），否则模型无法拟合非原点
        x = np.column_stack([x, np.ones(len(x))])
        self.m, self.n = x.shape
        self.mask = np.eye(self.k_nums)[y]
        w = np.zeros((self.n, self.k_nums))
        
        for i in range(self.max_iters): # 牛顿法收敛极快
            grad = self.first_grad(x, w)
            hessian = self.second_grad(x, w)
            
            try:
                # 计算二阶导数的你矩阵
                delta = solve(hessian, grad.reshape(-1, 1))
                new_w = w - self.lr * delta.reshape(self.n, self.k_nums)
                
                '''检查梯度模长是否接近 0 (更精准的收敛判断)
                因为 有加入了l2正则 softmax函数变成严格凸函数 
                1 有且仅有一个全局最小值点
                2 最小值点的梯度为0 (衡量梯度值 用模长)
                '''
                if norm(grad) < 1e-5:
                    print(f"Converged at iteration {i}")
                    break
                w = new_w
            except np.linalg.LinAlgError:
                print("Hessian singular, stopping.")
                break
        
        self.w = w
        return self
    
    def predict_proba(self, test):
        # 预测时也需要加上偏置项
        test = np.column_stack([test, np.ones(test.shape[0])])
        proba=self._softmax(test,self.w)
        return proba
    
    def predict(self,test):
        proba=self.predict_proba(test)
        return proba.argmax(axis=1)
                
def main():
    x,y=load_wine(return_X_y=True)
    # x,y=load_iris(return_X_y=True)
    x=np.asarray(x)

    scaler=MinMaxScaler()
    clf=Logic(decay=0.01,max_iters=50)

    model=Pipeline([('scaler',scaler),('clf',clf)])

    cv_style=StratifiedKFold(5,shuffle=True)
    results=cross_validate(model,x,y,cv=cv_style)
    print(f"{results['test_score']}")


if __name__ == '__main__':
    main()


