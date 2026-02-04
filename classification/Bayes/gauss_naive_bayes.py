import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm
import pandas as pd

from sklearn.datasets import load_iris,load_wine,load_digits
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.model_selection import cross_validate,StratifiedKFold
from sklearn.naive_bayes import GaussianNB


class GaussBayes(ClassifierMixin,BaseEstimator):

    def __init__(self,var_smothing:float=1e-5):
        self.var_smothing=var_smothing #增大平滑指数 对某些数据集提升明显

    def _softmax(self,x:NDArray):
        '''
        使用这个函数并不会影响预测的结果 这是尽可能满足继承的两个类的接口 要保证
        输出的类别概率和=1 
        '''
        exp=np.exp(x-x.max(axis=1,keepdims=True))
        return exp/(exp.sum(axis=1,keepdims=True))
    
    def fit(self,x:NDArray,y:NDArray):

        self.classes_,nums=np.unique(y,return_counts=True)# 统计标签 用于计算先验概率
        self.log_y=np.log(nums/nums.sum()) #计算先验概率 并转对疏

        df=pd.DataFrame(x) 
        groups:pd.DataFrame=df.groupby(y).agg(['mean','var'])# 按列 计算各列的均值和方差
        self.groups=groups.swaplevel(0,1,axis=1).sort_index(level=0,axis=1)#交换列索引 并对外层索引排序
        
        return self
    
    def _cal_prob(self,df,test:NDArray):

        '''groupby 分组函数  计算各个类别下的 每个样本 每个特征的对数概率'''

        mean=df.loc[:,('mean')].values #获取 每列的均值
        var=df.loc[:,('var')].values #获取每列的方差
        std=np.sqrt(var+self.var_smothing)#这一步大幅提高digits数据集中模型的效果
        '''norm.logpdf 可以直接test的pdf概率 logpdf是直接转log'''

        p=norm.logpdf(test,mean,std) # (m,n_features) 返回的是test中每个样本 对应的各个特征下的概率
        sum_p=p.sum(axis=1)# 对数概率可以直接相加 (m,)
        return pd.Series(sum_p)  #必须返回series 这样才能保证 groupby返回的是dataframe
    
  
    # def predict_proba(self,X):
          #常规实现方式  
    #     X=np.atleast_2d(X)
    #     cond_proba=self.groups.groupby(level=0).apply(self._cal_prob,X) #(num_class,m)Datafram 
    #     cond_proba=cond_proba.to_numpy()
    #     joint_proba=(cond_proba+self.log_y[:,None]).T #  广播加法运算 计算后验概率
    #     softmax_proba=self._softmax(joint_proba)
    #     return softmax_proba

    def predict_proba(self,X): # X (m,r)

        '''
        向量化实现 快速推导 避免 pandas 开销
        '''

        X=np.atleast_2d(X)[None,:,:]  #(1,m,r)
        means=self.groups.loc[:,('mean')].to_numpy()[:,None,:] #(k,1,r)
        vars=(self.groups.loc[:,'var']+self.var_smothing).to_numpy()[:,None,:] #(k,1,r)

        cond_proba=-1/2*np.log(2*np.pi*vars)+(-(X-means)**2/(2*vars)) #高斯概率密度函数 (k,m,r)
        joint_proba=(cond_proba.sum(axis=2))+self.log_y[:,None] # (k,m)
        softmax_logproba=self._softmax(joint_proba.T)

        return softmax_logproba
        
    
    def predict(self,X):

        proba=self.predict_proba(X)
        label=np.argmax(proba,axis=1)
        return self.classes_[label]


def main():

    x,y=load_digits(return_X_y=True)
    x,y=load_iris(return_X_y=True)
    x=np.asarray(x)
    model=GaussBayes(1)
    # model=GaussianNB()

    cv_style=StratifiedKFold(5,shuffle=True)
    
    results=cross_validate(model,x,y,cv=cv_style)

    print(f"acc: {results['test_score'].mean():.4f}")



if __name__ == '__main__':
    main()
