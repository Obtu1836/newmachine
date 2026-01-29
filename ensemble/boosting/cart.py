import numpy as np
from numpy.typing import NDArray
import sys

from sklearn.datasets import make_regression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split

class Node:
    def __init__(self, column=None, value=None, left=None,
                 right=None, leaf=None):

        self.column = column
        self.value = value
        self.left = left
        self.right = right
        self.leaf = leaf


class Cart:
    def __init__(self, min_samples_leaf: int, max_depth: int = sys.maxsize,
                       min_samples_split:int=5,quan_size:int=10):

        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_samples_split=min_samples_split
        self.quansize=quan_size

    def _split(self, x: NDArray, y: NDArray, col: int, val: float):
        mask = x[:, col] < val
        return x[mask], y[mask], x[~mask], y[~mask]
    
    def _get_best_params(self,x:NDArray,y:NDArray):

        increase=0
        params={}

        init_var=np.var(y)
        _,n=x.shape
        for col in range(n):
            features=np.unique(x[:,col])
            if len(features)>self.quansize:
                feature=np.quantile(features,np.linspace(0.01,0.99,self.quansize))
            else:
                feature=features
            for val in feature:
                l_x,l_y,r_x,r_y=self._split(x,y,col,val)
                if len(l_y)<=self.min_samples_leaf or len(r_y)<=self.min_samples_leaf:
                    continue
                
                l_var=np.var(l_y)
                r_var=np.var(r_y)

                '''下面这一步计算长度用来加权的 防止子树的var过大或过小'''
                new_var=(l_var*len(l_x)+r_var*len(r_x))/len(x)
                diff=init_var-new_var
                if diff>increase:
                    increase=diff
                    params.update({'l_x':l_x,'r_x':r_x,
                                   'column':col,'value':val,
                                   'l_y':l_y,'r_y':r_y})
        return params
    
    def _build_tree(self,x:NDArray,y:NDArray,depth:int):

        if depth<=0:
            return Node(leaf=y.mean())
        
        if len(x)<=self.min_samples_split:
            return Node(leaf=np.mean(y))
        
        params=self._get_best_params(x,y)
        if not params:
            return Node(leaf=np.mean(y))
        
        left=self._build_tree(params['l_x'],params['l_y'],depth-1)
        right=self._build_tree(params['r_x'],params['r_y'],depth-1)

        return Node(column=params['column'],value=params['value'],
                    left=left,right=right)
    
    def fit(self,x,y):

        self.node=self._build_tree(x,y,self.max_depth)
        return self
    
    def _predict(self,node:Node,test:NDArray):

        if node.leaf is not None:
            return node.leaf
        
        if test[node.column]<node.value:
            return self._predict(node.left,test) #type: ignore
        return self._predict(node.right,test) # type: ignore
    
    def predict(self,test):

        yp=[self._predict(self.node,t) for t in test]
        return np.array(yp)
    

def main():

    x,y=make_regression(1200,3)[:2]

    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)

    cart=Cart(5)
    cart.fit(x_train,y_train)

    yp=cart.predict(x_test)

    print(f'R2: {r2_score(y_test,yp)}')
    print(f'mse: {mean_squared_error(y_test,yp)}')

if __name__ == '__main__':
    main()




        

        
        
        