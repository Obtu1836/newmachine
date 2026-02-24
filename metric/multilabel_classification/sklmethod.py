import numpy as np 

from sklearn.model_selection import cross_validate,KFold
from sklearn.datasets import make_multilabel_classification
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer,jaccard_score

clf=ClassifierChain(LogisticRegression(max_iter=5000,l1_ratio=0))
scaler=StandardScaler()
model=Pipeline([('scaler',scaler),('clf',clf)])

def main():
    x,y,_,_=make_multilabel_classification(1200,8,return_distributions=True)

    jacc=make_scorer(jaccard_score,average='samples',zero_division=0)
    cv_style=KFold(5,shuffle=True)
    score={'jacc':jacc}
    results=cross_validate(model,x,y,cv=cv_style,scoring=score)

    print(results['test_jacc'].mean())

if __name__ == '__main__':
    main()


