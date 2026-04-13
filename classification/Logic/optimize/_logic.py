import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin,BaseEstimator
from scipy.special import softmax,log_softmax
from .optimizer import create_optimizer

from collections import deque


class Logic(ClassifierMixin,BaseEstimator):
    def __init__(self,lr:float,
                      optimizer_name:str|None=None,
                      beta1=0.9,
                      beta2=0.95,
                      iters:int=200000):
        
        self.lr=lr
        self.optimizer_name=optimizer_name
        self.beta1=beta1
        self.beta2=beta2
        self.iters=iters

        self.max_length=10

    def fit(self,x,y):

        self.optimizer_=None
        self.losses=deque(maxlen=self.max_length)
        self.X=np.column_stack([x,np.ones(len(x))])
        self.m,n=self.X.shape
        self.classes_=np.unique(y)
        self.num_class=len(self.classes_)

        self.mask=np.eye(self.num_class)[y]
        self.w_=np.random.randn(n,self.num_class)

        initv,initr=np.zeros_like(self.w_),np.zeros_like(self.w_)
        if self.optimizer_name is not None:
            self.optimizer_=create_optimizer(self.optimizer_name,
                                            initv=initv,initr=initr,
                                            beta1=self.beta1,beta2=self.beta2)
        epochs=0
        while True:
            grad=self._calculte_grad()
            if self.optimizer_:
                m,r=self.optimizer_.step(grad)
                if r is None:
                    r=1
            else:
                m,r=grad,1
            self.w_=self.w_-(self.lr/(np.sqrt(r)+1e-8))*m
            loss=self._calculate_loss()
            self.losses.append(loss)
            if len(self.losses)>=self.max_length:
                avg=np.array(self.losses).mean()
                if abs(avg-loss)<1e-5:
                    print(f'{epochs} iters fit over')
                    break
            epochs+=1
            if epochs>self.iters:
                print('exceeded_the_maximum_number_of_iterations model_does_not_converge')
                break
        return self

    def _calculte_grad(self):
        grad=self.X.T.dot(softmax(self.X.dot(self.w_),axis=1)-self.mask)
        return grad/self.m
    
    def _calculate_loss(self):
        loss=(self.mask*(log_softmax(self.X.dot(self.w_),axis=1))).sum()/self.m
        return loss

    def predict_proba(self,test):
        test=np.column_stack([test,np.ones(len(test))])
        probas=softmax(test.dot(self.w_),axis=1)
        
        return probas

    def predict(self,test):
        probas=self.predict_proba(test)
        label=np.argmax(probas,axis=1)
        return label

def main():
    ALL_OPTIMIZER_NAME=[None,'SGDMomentum', 'EMAMomentum', 'AdaGrad', 'RMSProp', 'Adam']

    x,y=load_iris(return_X_y=True)
    x=np.asarray(x)
    clf=Logic(1e-1,ALL_OPTIMIZER_NAME[0],iters=200000)
    scaler=MinMaxScaler()

    model=Pipeline([('scaler',scaler),('clf',clf)])

    cv_style=StratifiedKFold(5,shuffle=True,random_state=10)
    score=cross_validate(model,x,y,cv=cv_style)
    print(score['test_score'])

if __name__ == '__main__':
    main()


