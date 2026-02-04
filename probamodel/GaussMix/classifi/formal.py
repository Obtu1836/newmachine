import numpy as np 
from numpy.typing import NDArray

from scipy.special import logsumexp
from ..cluster.formal import GaussMixture
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator,ClassifierMixin

class Model(GaussMixture):

    def predict_proba(self,test:NDArray):

        test=np.atleast_2d(test)
        proba=super().predict_proba(test)
        logsum_proba=logsumexp(proba,axis=1)

        return logsum_proba

class ClassifiModel(BaseEstimator,ClassifierMixin):

    def __init__(self,k:int,alpha:float,max_iter:int=2000):

        self.k=k
        self.alpha=alpha
        self.max_iter=max_iter
        self.models:dict[int,Model]={}
    
    def fit(self,X,y):

        self.classes_,nums=np.unique(y,return_counts=True)
        self.k_class=len(self.classes_)

        self.log_y_proba=np.log(nums/nums.sum())

        for c in self.classes_:
            x_c=X[y==c]
            model=Model(self.k,max_iters=self.max_iter,
                         alpha=self.alpha)
            model.fit(x_c)
            self.models[c]=model

        return self

    def predict_proba(self,test):

        probas=np.zeros((len(test),self.k_class))
        for cls,model in self.models.items():
            proba=model.predict_proba(test)
            probas[:,cls]=proba
        probas+=self.log_y_proba
        return probas
    

    def predict(self,test):

        probas=self.predict_proba(test)
        label=probas.argmax(axis=1)
        return label

    

def main():
    x,y=load_iris(return_X_y=True)
    x=np.asarray(x)

    scaler=MinMaxScaler()
    clf=ClassifiModel(3,1e-2)

    model=Pipeline([('scaler',scaler),('clf',clf)])

    cv_style=StratifiedKFold(5,shuffle=True)
    results=cross_val_score(model,x,y,cv=cv_style)
    print(f"{'mean':<8} {results.mean():.3f} \
          {'std':<5} +-{results.std():.2f}")

if __name__ == '__main__':
    main()




    




