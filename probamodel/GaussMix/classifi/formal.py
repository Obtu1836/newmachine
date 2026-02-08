import numpy as np 
from scipy.stats import multivariate_normal
from scipy.special import logsumexp,softmax

from sklearn.datasets import load_iris,load_digits,load_wine
from sklearn.model_selection import cross_validate,StratifiedKFold  
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer,recall_score,precision_score

'''
使用高斯混合模型进行分类任务
对样本的每个类别 分别使用k个高斯分布拟合 总共需要 N*K个高斯分布 N为类别数
每一个高斯混合模型类 使用k个高斯分布 拟合一个类别 最后将k个高斯分布的概率合并为一个概率
最终 每个类别 都有一个概率 比较最大值 
'''

class GaussMinClassfi:

    def __init__(self,k_nums:int,*,max_iters:
                 int=2000,rtol:float=1e-6,alpha:float=1e-3):
        
        self.k_nums=k_nums
        self.alpha=alpha
        self.rtol=rtol
        self.max_iters=max_iters
    
    def _roulette(self,data):#轮盘法 确定高斯分布的均值
        cents=[]
        cents.append(data[np.random.choice(len(data))])

        while len(cents)<self.k_nums:
            dis2=np.sum(np.power(data[:,None]-cents,2),axis=2)
            mindis2=np.min(dis2,axis=1)
            dis_prob=mindis2/mindis2.sum()
            prob_cum=np.cumsum(dis_prob)
            idx=np.searchsorted(prob_cum,np.random.rand())
            cents.append(data[idx])
        
        return np.array(cents)

    def _init_params(self,data):

        mean=self._roulette(data)
        #初始的协方差 为样本的协方差
        '''
        等价于  dx=data-mean(axis=0)
               cov=np.dot(dx.T,dx)/(len(data)-1)  无偏估计
        '''
        # cov=np.array([np.cov(data.T) for i in range(self.k_nums)])
        cv=np.cov(data.T)
        cov=np.tile(cv,(self.k_nums,1,1))
        weight=np.random.rand(self.k_nums) #权重 随机设置 需要归一化
        weight/=weight.sum()

        return mean,cov,weight

    def fit(self,x):
        
        self.m,self.n=x.shape
        self.alpha=np.eye(self.n)*self.alpha
        self.data=x
        mean,cov,weight=self._init_params(x)

        init_loss=np.inf
        for i in range(self.max_iters):
            hidden,loss=self.e_step(mean,cov,weight)
            new_mean,new_cov,new_weight=self.m_step(hidden)
            if abs(init_loss-loss)<self.rtol:#当总的变化量小于阈值 停止迭代
                # print(f'break in {i} loops')
                break
            mean,cov,weight=new_mean,new_cov,new_weight
            init_loss=loss
        else:
            print('max_iters maybe not enough')
        
        self.mean=new_mean
        self.cov=new_cov
        self.weight=new_weight

        return self
    
    def e_step(self,mean,cov,weight):

        '''
        根据初始的均值 协方差  和权重 计算隐变量
        '''

        hidden=np.zeros((self.m,self.k_nums))
        for i in range(self.k_nums):
            mulmal=multivariate_normal(mean[i],cov[i]+self.alpha,allow_singular=True)
            hidden[:,i]=mulmal.logpdf(self.data)
        
        log_hidden=hidden+np.log(weight+self.rtol)
        log_hidden_axis=logsumexp(log_hidden,axis=1)
        log_hidden_axis=np.asarray(log_hidden_axis)
        log_loss=log_hidden_axis.sum()

        return np.exp(log_hidden-log_hidden_axis[:,None]),log_loss
    
    def m_step(self,hidden):

        '''
        根据隐变量 更新 均值 协方差和权重 
        '''
        hz=hidden.sum(axis=0)

        #hidden的每一列 分别与data相乘 并累加 最后除以每列的'总数' 
        mean=(hidden[...,None]*self.data[:,None,:]).sum(axis=0)
        mean/=hz[:,None]

        '''
        更新 协方差时 使用np.dot 属于手动更新 因为需要利用新的均值和
        hidden 
        '''
        # cov=np.zeros((self.k_nums,self.n,self.n))
        # for i in range(self.k_nums):
        #     dx=self.data-mean[i]
        #     p=np.dot(dx.T*hidden[:,i],dx)
        #     cov[i]=p/hz[i]
        # 向量化版本实现
        mkn=self.data[:,None]-mean

        mknw=mkn*hidden[...,None]
        knmw=np.transpose(mknw,(1,2,0))
        kmn=np.transpose(mkn,(1,0,2))

        cov=(knmw@kmn)/hz[:,None,None]
        
        weight=hz/self.m

        return mean,cov,weight
    
    def predict_proba(self,test):

        test=np.atleast_2d(test)
        hidden=np.zeros((len(test),self.k_nums))
        for i in range(self.k_nums):
            mulmal=multivariate_normal(self.mean[i],self.cov[i]+self.alpha,allow_singular=True)
            hidden[:,i]=mulmal.logpdf(test)
        
        log_hidden=hidden+np.log(self.weight+self.rtol)
        log_hidden_axis=logsumexp(log_hidden,axis=1)

        return log_hidden_axis
    
class Model(ClassifierMixin,BaseEstimator):
    
    def __init__(self,k_nums:int,max_iters:int=2000,
                 rtol:float=1e-6,alpha=1e-3):
        
        self.k_nums=k_nums
        self.alpha=alpha
        self.rtol=rtol
        self.max_iters=max_iters

    def fit(self,x,y):

        self.classes_,nums=np.unique(y,return_counts=True)
        self.logy=np.log(nums/nums.sum()+self.rtol)

        self.dic:dict[int|str,GaussMinClassfi]={}
        for i in self.classes_:
            clf=GaussMinClassfi(self.k_nums,max_iters=self.max_iters,
                                 rtol=self.rtol,alpha=self.alpha)
            x_=x[y==i]
            clf.fit(x_)
            self.dic[i]=clf
        
        return self
    
    def predict_proba(self,test):

        test=np.atleast_2d(test)
        probas=np.zeros((len(test),len(self.classes_)))
        for idx,k in enumerate(self.classes_):
            clf=self.dic[k]
            probas[:,idx]=clf.predict_proba(test)
        probas+=self.logy # 加上先验概率 才是完整的贝叶斯
        return softmax(probas,axis=1)
    
    def predict(self,test):

        probas=self.predict_proba(test)
        return probas.argmax(axis=1)

def main():
    # x,y=load_iris(return_X_y=True)
    x,y=load_digits(return_X_y=True)
    x=np.asarray(x)

    scaler=MinMaxScaler()
    clf=Model(2)

    model=Pipeline([('scaler',scaler),('clf',clf)])

    cv_style=StratifiedKFold(5,shuffle=True)
    
    recall=make_scorer(recall_score,average='macro')
    precision=make_scorer(precision_score,average='macro')

    score={'acc':'accuracy',
           'recall':recall,
           'precision':precision}

    resuluts=cross_validate(model,x,y,scoring=score,cv=cv_style)

    print(f"{'acc':<12}{resuluts['test_acc'].mean():.3f}")
    print(f"{'recall':<12}{resuluts['test_recall'].mean():.3f}")
    print(f"{'precision':<12}{resuluts['test_precision'].mean():.3f}")


if __name__ == '__main__':
    main()

    





    



    




        



