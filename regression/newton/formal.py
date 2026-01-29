import numpy as np
from numpy.linalg import inv
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

def first_derivative(x, w, y):
    return x.T.dot(x.dot(w) - y)

def second_derivative(x):
    return x.T.dot(x)

def fun(x, y):
    y = y[:, None]
    _, n = x.shape
    w = np.zeros((n, 1))

    while True:
        neww=w-inv(second_derivative(x)).dot(first_derivative(x,w,y))
        if np.allclose(neww,w):
            break
        w=neww
    return neww

def main():
    x,y=make_regression(250,4,noise=0.1)[:2]
    print(x.shape)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_train=np.c_[x_train,np.ones_like(y_train)]
    x_test = scaler.transform(x_test) 
    x_test=np.c_[x_test,np.ones_like(y_test)]


    w = fun(x_train, y_train)
    y_pred = x_test.dot(w).ravel()
    print(f"R2 Score: {r2_score(y_test, y_pred)}")

if __name__ == '__main__':
    main()


