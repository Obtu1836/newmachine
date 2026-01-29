import numpy as np 
from collections import deque

from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import ClassifierMixin,BaseEstimator

class LogicSoftmax(ClassifierMixin,BaseEstimator):

    def __init__(self,lr:float,weight):

        self.lr=lr
        self.weight=weight
    
    def loss(self,prob,mask,weights):
        return -(mask*np.log(prob)*weights).sum()/len(mask)

    def softmax(self,x,w):
        x=x.dot(w)
        exp=np.exp(x-np.max(x,axis=1,keepdims=True)) # 分子分母同除最大值 值不变 且防止数值溢出
        return exp/exp.sum(axis=1,keepdims=True)
    
    def fit(self,x,y):
        self.classes_,nums=np.unique(y,return_counts=True)
        k=len(self.classes_)
        mask=np.eye(k)[y]
        x=np.column_stack([x,np.ones(len(x))])
        y=y[:,None]
        m,n=x.shape
        w=np.zeros((n,k))
        deq_loss=deque(maxlen=5)

        reciprocal=1/nums
        if self.weight:
            weights=reciprocal/reciprocal.min()[y]
        else:
            weights=np.ones((m,1))

        while True:
            grad=(-x.T.dot((mask-self.softmax(x,w))*weights))/m
            w-=self.lr*grad
            new_loss=self.loss(self.softmax(x,w),mask,weights)
            deq_loss.append(new_loss)

            if len(deq_loss)>=5:
                d_los=np.array(deq_loss)
                if np.allclose(d_los,d_los.mean()):
                    break
        
        self.w_=w
        return self
    
    
    def predict_proba(self,test):
        test=np.column_stack([test,np.ones(len(test))])
        proba=self.softmax(test,self.w_)
        return proba
    
    def predict(self,test):
        proba=self.predict_proba(test)
        y_pred=np.argmax(proba,axis=1)

        return y_pred

def main():
    x,y=load_iris(return_X_y=True)
    x=np.array(x)

    scaler=MinMaxScaler()
    clf=LogicSoftmax(0.01,False)
    model=Pipeline([('scaler',scaler),('clf',clf)])

    scors=['accuracy','precision_macro','recall_macro']
    result=cross_validate(model,x,y,cv=5,scoring=scors)

    print(result['test_accuracy'].mean(),
          result['test_precision_macro'].mean())
    
if __name__ == '__main__':
    main()




        

    