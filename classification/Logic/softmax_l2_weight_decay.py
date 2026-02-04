import numpy as np 
from numpy.linalg import solve,norm
from sklearn.datasets import load_wine
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.model_selection import cross_validate,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

class Logic(BaseEstimator,ClassifierMixin):

    def __init__(self,lambd=0.1,lr=0.5):

        self.lambd=lambd # 强正则化，解决 Softmax 参数冗余问题
        self.lr=lr # 阻尼牛顿法，防止初期更新过猛导致不收敛
    
    def _softmax(self,x,w):

        x=x.dot(w)
        exp=np.exp(x-x.max(axis=1,keepdims=True))
        return exp/exp.sum(axis=1,keepdims=True)

    def _loss(self, x, w, mask):
        m = x.shape[0]
        # 计算原始交叉熵损失
        probs = self._softmax(x, w)
        # 加入 1e-12 这种微小值防止 log(0) 报错
        cross_entropy = -np.sum(mask * np.log(probs + 1e-12)) / m
        
        # 增加 L2 正则项: (lambda / 2) * sum(w^2)
        # 通常不惩罚偏置项，但如果把偏置合并在 w 里了，全写进去也行
        # 正则项就是 np.sum(np.power(w,2))
        l2_penalty = 0.5 * self.lambd * np.sum(w**2)
        
        return cross_entropy + l2_penalty

    def first_grad(self, x, w, mask, ):
        # 增加 L2 正则化的梯度: grad + lambda * w
        m = x.shape[0]
        grad = -x.T.dot(mask - self._softmax(x, w)) / m
        #正则项的导数 就是本身  计算导数的过程就是逐元素求导(开方)
        return grad + self.lambd * w 
    
    def second_grad(self, x, w):
        n, k = w.shape
        m = x.shape[0]
        hessian = np.zeros((n * k, n * k))
        probs = self._softmax(x, w)
        for i in range(m):
            xi_xt = np.outer(x[i], x[i])
            pi = probs[i]
            s_i = np.diag(pi) - np.outer(pi, pi)
            hessian += np.kron(xi_xt, s_i)
        
        #  Hessian 也要加上 L2 正则项的二阶导
        return (hessian / m) + self.lambd * np.eye(n * k)

    def fit(self, x, y):
        self.classes_ = np.unique(y)
        k_nums = len(self.classes_)
        #  增加偏置项（Intercept），否则模型无法拟合非原点
        x = np.column_stack([x, np.ones(x.shape[0])])
        m, n = x.shape
        mask = np.eye(k_nums)[y]
        w = np.zeros((n, k_nums))
        
       
        for i in range(50): # 牛顿法收敛极快
            grad = self.first_grad(x, w, mask)
            hessian = self.second_grad(x, w,)
            
            try:
                # 计算二阶导数的你矩阵
                delta = solve(hessian, grad.reshape(-1, 1))
                new_w = w - self.lr * delta.reshape(n, k_nums)
                
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
        logits = test.dot(self.w)
        return logits 
    
    def predict(self,test):
        proba=self.predict_proba(test)
        return proba.argmax(axis=1)
                
def main():
    x,y=load_wine(return_X_y=True)
    x=np.asarray(x)

    scaler=MinMaxScaler()
    clf=Logic()

    model=Pipeline([('scaler',scaler),('clf',clf)])

    cv_style=StratifiedKFold(5,shuffle=True)
    results=cross_validate(model,x,y,cv=cv_style)
    print(f"{results['test_score']}")


if __name__ == '__main__':
    main()


