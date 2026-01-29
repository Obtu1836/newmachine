import numpy as np
from numpy.typing import NDArray
from typing import Literal
from dataclasses import dataclass

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@dataclass
class Node:
    column: int | None = None
    value: float | None = None
    left: "Node|None" = None
    right: "Node|None" = None
    leaf: int | None = None


class DecisionTree:

    def __init__(self, max_depth: int, mode: Literal['entropy', 'gini']):

        self.max_depth = max_depth
        self.mode = mode

    def _calculate_metric(self, label) ->float:

        if len(label) <= 0:
            return 0
        _, nums = np.unique(label, return_counts=True)
        p = nums/nums.sum()

        if self.mode == 'entropy':
            return np.sum(-p*np.log2(p))
        return 1-np.power(p, 2).sum()

    def _split(self, data: NDArray, label: NDArray, col: int, value: float):

        mask = data[:, col] < value
        return data[mask], label[mask], data[~mask], label[~mask]

    def _get_split_param(self, data: NDArray, label: NDArray) ->dict:

        increase = 0
        params = {}
        init_metric = self._calculate_metric(label)

        _, n = data.shape
        for col in range(n):
            values = np.unique(data[:, col])
            for val in values:
                l_data, l_label, r_data, r_label = self._split(
                    data, label, col, val)
                if len(l_data) == 0 or len(r_data) == 0:
                    continue
                l_metric = self._calculate_metric(l_label)
                r_metric = self._calculate_metric(r_label)

                new_metric = (l_metric*len(l_data) +
                              r_metric*len(r_data))/len(data)

                diff = init_metric-new_metric
                if diff > increase:
                    increase = diff
                    params.update({'column': col, 'value': val,
                                   'l_data': l_data, 'r_data': r_data,
                                   'l_label': l_label, 'r_label': r_label})
        return params
    
    def _cal_leaf(self,label:NDArray) -> int:

        lab,nums=np.unique(label,return_counts=True)
        tag=lab[np.argmax(nums)]
        return tag

    def _build_tree(self, data: NDArray, label: NDArray, depth: int) ->Node:

        if len(np.unique(label)) == 1:
            return Node(leaf=label[0])
        
        if depth>=self.max_depth:
            return Node(leaf=self._cal_leaf(label))
        
        params=self._get_split_param(data,label)
        if not params:
            return Node(leaf=self._cal_leaf(label))
        
        left=self._build_tree(params['l_data'],params['l_label'],depth+1)
        right=self._build_tree(params['r_data'],params['r_label'],depth+1)

        return Node(column=params['column'],value=params['value'],
                    left=left,right=right)
    
    def fit(self,data:NDArray,label:NDArray):

        self.tree=self._build_tree(data,label,0)

    def predict(self,tests:NDArray):

        result=np.array([self._predict_x(self.tree,test) for test in tests])
        return result

    def _predict_x(self,tree:Node,test:NDArray):

        if tree.leaf is not None:
            return tree.leaf
        
        if test[tree.column]<tree.value:
            return self._predict_x(tree.left,test) # type: ignore
        return self._predict_x(tree.right,test)  # type: ignore
    
    def print_path(self,prefix='ROOT'):
        
        def inner(tree:Node,level:str):

            if tree.leaf is not None:
                print(level+" leaf="+str(tree.leaf))
            else:
                print(level+'[ col: '+str(tree.column)+',val: '+str(tree.value)+']')
                if tree.left is not None and tree.right is not None:
                    inner(tree.left,level+'-L')
                    inner(tree.right,level+'-R')

        inner(self.tree,prefix)



def main():
    x,y=load_iris(return_X_y=True)
    trainx,testx,trainy,testy=train_test_split(x,y,train_size=0.8)

    desc=DecisionTree(5,'entropy')

    desc.fit(trainx,trainy)

    yp=desc.predict(testx)

    print(f"acc: {accuracy_score(testy,yp)}")

    desc.print_path()

if __name__ == '__main__':
    main()



    

