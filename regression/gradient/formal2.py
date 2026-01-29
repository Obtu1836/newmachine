import numpy as np
from sklearn.datasets import make_regression,load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


def grad(x,w,y):
    return x.T.dot(x.dot(w)-y)

def gradient_descent(x,y):
    
    _,n=x.shape
    w=np.zeros((n,1))
    lr=0.001
    while True:
        neww=w-lr*grad(x,w,y)
        if np.allclose(neww,w):
            break
        w=neww
    return neww


def main():

    x,y=make_regression(450,4,noise=0.1)[:2]
    train_x,test_x,train_y,test_y=train_test_split(x,y,train_size=0.8)

    scaler=StandardScaler()
    train_x=scaler.fit_transform(train_x)
    train_x=np.c_[train_x,np.ones_like(train_y)]

    test_x=scaler.transform(test_x)
    test_x=np.c_[test_x,np.ones_like(test_y)]

    w=gradient_descent(train_x,train_y[:,None])
    print(w)

    yp=test_x.dot(w)

    print(f'r2_score: {r2_score(test_y,yp.ravel())}')



if __name__ == '__main__':
    main()


