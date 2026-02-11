import numpy as np 
from numpy.typing import NDArray

from collections import deque

from sklearn.datasets import load_breast_cancer
from sklearn.base import ClassifierMixin,BaseEstimator
from sklearn.metrics import make_scorer,recall_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

class Logic(ClassifierMixin,BaseEstimator):

    def __init__(self,lr:float,weight:bool=False):

        self.lr=lr
        self.weight=weight
    
    def sigmoid(self,x:NDArray,w:NDArray):
        xw=np.clip(-x.dot(w),-250,250)
        return (1/(1+np.exp(xw)))
    
    def loss(self,x:NDArray,w:NDArray,y,weight,):
        m=x.shape[0]
        return -(weight*(y*x.dot(w)+np.log(1-self.sigmoid(x,w)))).sum()/m
    
    def fit(self,x,y):
        x=np.column_stack([x,np.ones(len(x))])
        y=y[:,None]
        self.classes_,nums=np.unique(y,return_counts=True)
        loss_que=deque(maxlen=5)
        m,n=x.shape
        w=np.zeros((n,1))

        if self.weight:
            reciprocal=1/nums
            weight=(reciprocal/reciprocal.min())[y]
        else:
            weight=np.ones_like(y)

        while True:
            grad=(-x.T.dot((y-self.sigmoid(x,w))*weight))/m
            w-=self.lr*grad
            new_loss=self.loss(x,w,y,weight)
            loss_que.append(new_loss)
            if len(loss_que)==5:
                loss_arr=np.array(loss_que)
                if np.allclose(loss_arr,loss_arr.mean(),rtol=1e-5):
                    break
        self.w_=w

        return self
    
    def predict_proba(self,test):
        test=np.column_stack([test,np.ones(len(test))])
        prob=self.sigmoid(test,self.w_)
        prob=np.column_stack([1-prob,prob])
        return prob
    
    def predict(self,test):
        prob=self.predict_proba(test)
        ypred=np.where(prob[:,1]>0.5,1,0)
        return ypred

def main():
    x,y=load_breast_cancer(return_X_y=True)
    x=np.asarray(x)

    scaler=MinMaxScaler()
    clf=Logic(0.01,True)

    model=Pipeline([('scaler',scaler),('clf',clf)])

    recall_neg=make_scorer(recall_score,pos_label=0)

    score={'recall_neg':recall_neg,
           'recall_pos':'recall',
           'precision':'precision',
           'acc':'accuracy'}

    result=cross_validate(model,x,y,cv=5,scoring=score)
    print(result['test_recall_pos'].mean())
    print(result['test_recall_neg'].mean())
    print(result['test_precision'].mean(),
          result['test_acc'].mean())

if __name__ == '__main__':
    main()





    

        


    




