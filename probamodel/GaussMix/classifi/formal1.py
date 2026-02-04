import numpy as np 
from sklearn.datasets import load_digits,load_iris,load_wine
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


class GaussMix:

    def __init__(self,k:int,alpha:float,max_iter:int=2000,tol:float=1e-6):

        self.k=k
        self.alpha=alpha
        self.max_iter=max_iter
        self.tol=tol

    def _plus(self,data):

        cents=[]
        cents.append(data[np.random.randint(len(data))])

        while len(cents)<self.k:
            dis2=np.sum(np.power(data[:,None]-cents,2),axis=2)
            mindis2=dis2.min(axis=1)
            dis_prob=mindis2/mindis2.sum()
            cum_prob=np.cumsum(dis_prob)
            idx=np.searchsorted(cum_prob,np.random.rand())
            cents.append(data[idx])
        
        return np.array(cents)

    def _init_param(self,data):

        mean = self._plus(data)
        
        init_cov=np.cov(data.T)+np.eye(self.n)*self.alpha
        cov=np.array([init_cov for i in range(self.k)]) #(k,n,n)

        weight=np.random.rand(self.k) #(k,)
        weight=weight/weight.sum()

        return mean,cov,weight

    def fit(self,data):
        
        self.data=data
        self.m,self.n=data.shape
        self.classes_=np.arange(self.k)
        mean,cov,weight=self._init_param(data)

        prev_likelihood = -np.inf

        for _ in range(self.max_iter): # 建议替换 while True 为带上限的循环
            z_hidden, current_likelihood = self.e_step(mean, cov, weight)
            
            # 检查收敛状态
            if abs(current_likelihood - prev_likelihood) < self.tol:
                break
            prev_likelihood = current_likelihood

            mean, cov, weight = self.m_step(z_hidden)
        
        self.mean, self.cov, self.weight = mean, cov, weight

        return self

    def e_step(self, mean, cov, weight):
        weighted_log_probs = np.zeros((self.m, self.k))
        for i in range(self.k):
            mulmal = multivariate_normal(mean[i], cov[i], allow_singular=True)
            weighted_log_probs[:, i] = mulmal.logpdf(self.data) 
        
        weighted_log_probs+=np.log(weight)
        # 使用 logsumexp 计算总对数似然
        log_prob_norm = logsumexp(weighted_log_probs, axis=1)
        log_prob_norm=np.asarray(log_prob_norm)
        log_likelihood = np.sum(log_prob_norm)
        
        # 计算隐变量 z_hidden (Posterior): exp(log_p - log_p_sum)
        z_hidden = np.exp(weighted_log_probs - log_prob_norm[:, None])
        return z_hidden, log_likelihood

    def m_step(self, z_hidden):

        expt_z=z_hidden.sum(axis=0)

        new_mean=(z_hidden[...,None]*self.data[:,None,:]).sum(axis=0)
        new_mean/=expt_z[:,None]

        new_cov=np.zeros((self.k,self.n,self.n))
        for i in range(self.k):
            dx=self.data-new_mean[i]
            cov=np.dot((dx.T*z_hidden[:,i]),dx)/(expt_z[i]+1e-8)
            new_cov[i]=cov+np.eye(self.n)*self.alpha
        
        new_weight = expt_z / self.m
        return new_mean, new_cov, new_weight
    
    def predict_proba(self, test):
        test = np.atleast_2d(test)
        m, _ = test.shape
        weighted_log_probs = np.zeros((m, self.k))
        for i in range(self.k):
            # 使用训练好的 self.mean, self.cov, self.weight
            mulmal = multivariate_normal(self.mean[i], self.cov[i], allow_singular=True)
            weighted_log_probs[:, i] = mulmal.logpdf(test) + np.log(self.weight[i] )

        # 返回总对数似然 log P(x | model_i)
        return logsumexp(weighted_log_probs, axis=1)

    
class GMMGenerativeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, ks=2, alpha=1e-3):
        self.ks = ks
        self.alpha = alpha
        self.models = {}
        self.priors = {}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / len(X)
            gmm = GaussMix(k=self.ks, alpha=self.alpha)
            gmm.fit(X_c)
            self.models[c] = gmm
        return self

    def predict(self, X):
        # 计算每个样本在每个类别下的 log-posterior
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for idx, c in enumerate(self.classes_):
            # log P(X|C) + log P(C)
            scores[:, idx] = self.models[c].predict_proba(X) + np.log(self.priors[c] + 1e-10)
        
        return self.classes_[np.argmax(scores, axis=1)]

# 使用交叉验证
from sklearn.model_selection import cross_val_score

def main():
    x, y = load_digits(return_X_y=True)
    x=np.asarray(x)
    # 标准化通常放在 Pipeline 中
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('gmm_clf', GMMGenerativeClassifier(ks=2, alpha=1e-3))
    ])

    # 5 折交叉验证
    cv_scores = cross_val_score(pipeline, x, y, cv=5)
    
    print(f"CV Scores: {cv_scores}")
    print(f"Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

if __name__ == '__main__':
    main()














