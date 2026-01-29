import numpy as np
from numpy.linalg import inv
from sklearn.linear_model import LinearRegression

def my_leastsq(data,label):
    data=np.c_[data,np.ones_like(data)]
    return inv(data.T.dot(data)).dot(data.T.dot(label))


def sklearn_leastsq(data,label):

    data=data[:,None]
    liner=LinearRegression()
    liner.fit(data,label)
    '''
    x 二维数组  不需要手动加上截距
    y 一维数组
    '''

    return liner
    
    

def main():

    data=np.array([15,20,25,30,35,40])
    label=np.array([136,140,155,160,157,175])

    my_w=my_leastsq(data,label)
    sklearn_w=sklearn_leastsq(data,label)
    print(my_w)
    print(sklearn_w.coef_,sklearn_w.intercept_)

if __name__ == '__main__':
    main()



