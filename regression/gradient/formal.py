import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

def grad_w(x:NDArray,w:NDArray,y:NDArray) ->NDArray:

    return x.T.dot(x.dot(w)-y)

def sgd(x:NDArray,y:NDArray):

    x=np.c_[x,np.ones_like(x)]
    m,n=x.shape
    y=y[:,None]
    w=np.zeros((n,1))
    lr=1e-5

    while True:
        neww=w-lr*grad_w(x,w,y)
        if np.allclose(neww,w):
            break
        w=neww
    
    return neww


def main():
    
    data,label=make_regression(80,1,noise=8)[:2]
    xmax,xmin=data[:,0].max(),data[:,0].min()

    w=sgd(data,label)
    xx=np.linspace(xmin,xmax,100)
    yy=np.c_[xx,np.ones_like(xx)].dot(w)
    
    plt.scatter(data[:,0],label,color='g')
    plt.plot(xx,yy.ravel(),color='r')
    plt.show()


if __name__ == '__main__':
    main()



