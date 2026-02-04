import numpy as np 
from numpy.typing import NDArray

from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class GaussMixture:
    def __init__(self,k:int,*,max_iters:int,alpha:float=1e-3,
                 rtol:float=1e-6):

        self.k=k
        self.alpha=alpha
        self.max_iters=max_iters
        self.rtol=rtol


    def _roulette(self,data:NDArray):

        cents=[]
        cents.append(data[np.random.choice(len(data))])

        while len(cents)<self.k:
            dis2=np.sum(np.power(data[:,None]-cents,2),axis=2)
            dis2min=np.min(dis2,axis=1)
            dis_prob=dis2min/dis2min.sum()
            prob_cum=np.cumsum(dis_prob)
            idx=np.searchsorted(prob_cum,np.random.rand())
            cents.append(data[idx])
        
        return np.array(cents)
    
    def _init_params(self,data:NDArray):

        mean=self._roulette(data)

        init_cov=np.cov(data.T)
        cov=np.array([init_cov for i in range(self.k)])

        init_weight=np.random.rand(self.k)
        weight=init_weight/init_weight.sum()

        return mean,cov,weight

    def fit(self,data:NDArray):

        self.m,self.n=data.shape
        self.data=data
        self.reg=np.eye(self.n)*self.alpha
        mean,cov,weight=self._init_params(data)

        init_loss=-np.inf
        for i in range(self.max_iters):
            hidden,loss=self.e_step(mean,cov,weight)
            if abs(loss-init_loss)<self.rtol:
                # print('end!!')
                break
            new_mean,new_cov,new_weight=self.m_step(hidden)
            init_loss=loss
            mean,cov,weight=new_mean,new_cov,new_weight
        
        self.mean=mean
        self.cov=cov
        self.weight=weight

        return self

    
    def e_step(self,mean,cov,weight):
        '''
        根据 均值 协方差 和权重 计算 隐变量
        1 根据 均值和方差 得到 多元分布的概率(转对数)
        2 将得到的概率与权重相加 得到加权的概率  因为都转成对数了 所以加法运算
        3 计算总的对数似然 这一步是用来判断终止迭代的
        4 隐变量归一化 (从公式得出 隐变量是不在log内的 所以用np.exp还原
          exp内的减法 实际上是除法 做的归一化

        '''

        rtol=1e-10
        log_probs=np.zeros((self.m,self.k)) 
        for i in range(self.k):
            mulmal=multivariate_normal(mean[i],cov[i],allow_singular=True)
            log_probs[:,i]=mulmal.logpdf(self.data)

        log_probs+=np.log(weight+rtol) #(m,k)
        log_axis_sum=np.asarray(logsumexp(log_probs,axis=1))#(m,)
        log_loss=log_axis_sum.sum()

        hidden=np.exp(log_probs-log_axis_sum[:,None])
        return hidden,log_loss

    def m_step(self,hidden:NDArray):
        '''
        隐变量 就是每个样本 分别在k个高斯分布的概率 (m,k)
        根据隐变量 更新 均值 协方差和权重
        1 计算隐变量中每个类别所占比例的期望总量 (均值和协方差 归一化时当分母)
        2 计算样本与每个高斯分布的概率再累加 再除以expt_z 相当于加权平均
        3 根据第k个新的均值 计算样本第k个的加权协方差 
        4 更新权重 直接除以样本数量  
        '''

        expt_z=hidden.sum(axis=0) #(k,)

        #         (m,k,1)        (m,1,n)->(m,k,n)-->(k,n)
        new_mean=(hidden[...,None]*self.data[:,None,:]).sum(axis=0)
        new_mean/=expt_z[:,None]

        new_cov=np.zeros((self.k,self.n,self.n))#(k,n,n)
        for i in range(self.k):
            dx=self.data-new_mean[i] #(m,n)
            cv=np.dot(dx.T*hidden[:,i],dx)/expt_z[i] #(n,n)
            new_cov[i]=cv+self.reg
        
        new_weight=expt_z/self.m

        return new_mean,new_cov,new_weight
    
    def predict_proba(self,X:NDArray)->NDArray:
        
        X=np.atleast_2d(X)
        m,_=X.shape
        log_probs=np.zeros((m,self.k)) 
        for i in range(self.k):
            mulmal=multivariate_normal(self.mean[i],self.cov[i],allow_singular=True)
            log_probs[:,i]=mulmal.logpdf(X)
        log_probs+=np.log(self.weight)

        # return np.asarray(logsumexp(log_probs, axis=1))
        return log_probs
    

def match_label(yt,yp):

    d=max(yt.max(),yp.max())+1
    w=np.zeros((d,d))
    np.add.at(w,(yt,yp),1)
    idt,idp=linear_sum_assignment(w.max()-w)
    mask=np.zeros(d,dtype=int)
    mask[idp]=idt
    yp[:]=mask[yp]

    return yp


def main():
    k=4
    x,y=make_blobs(2000,n_features=2,centers=k)[:2]
    fig,ax=plt.subplots(1,2,figsize=(10,6))

    model=GaussMixture(k,max_iters=200)
    model.fit(x)
    prob=model.predict_proba(x)
    yp=prob.argmax(axis=1)
    label=match_label(y,yp)

    ax[0].scatter(x[:,0],x[:,1],c=y,cmap='viridis',vmin=y.min(), vmax=y.max())
    ax[1].scatter(x[:,0],x[:,1],c=label,cmap='viridis',vmin=label.min(), vmax=label.max())

    plt.show()


if __name__ == '__main__':
    main()


        














