import numpy as np
from numpy.typing import NDArray
# from types import MethodType

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from classification.DecisionTree.non_prune import DecisionTree,Node


class New(DecisionTree):
    '''
    New 的 Docstring
    通过验证集 递归的方式进行后剪枝 
    1 如果 传入的数据为空 返回原节点
    2 如果节点是叶子点  返回原节点 
    3 模拟验证过程 将数据送入节点 处理左右子树 确保节点在尝试剪枝时 使用的都是
      能到达节点的验证数据
    4 在当前的节点 用预测验证数据给出的标签 直接用当前的最大数量的标签直接作为预测 分别
      和验证集的标签 计算指标 如果节点的指标不如直接计算最大数量给出的标签 就执行剪枝

    '''

    def prune(self,val_data,val_label):
        def inner(node:Node,val_data:NDArray,val_label:NDArray):

            if len(val_data)==0:
                return node
            
            if node.leaf is not None:
                return node
            
            mask=val_data[:,node.column]<node.value #验证集数据分流 确保子树流入的数据都是按划分依据流入的
            if node.left is not None and node.right is not None:
                node.left=inner(node.left,val_data[mask],val_label[mask])
                node.right=inner(node.right,val_data[~mask],val_label[~mask])
            
                yp=np.array([self._predict_x(node,data) for data in val_data])
                acc_before=accuracy_score(val_label,yp)

                tag=self._cal_leaf(val_label)

                acc_new=accuracy_score(val_label,np.full(len(val_label),tag))

                if acc_new>=acc_before: # 如果直接用数量最多的标签直接预测 比生成左右树更好 就
                    return Node(leaf=tag) #将该节点直接变为 叶子节点 标签定为数量做的的标签
            return node

        self.tree=inner(self.tree,val_data,val_label)

def main():
    x,y=load_iris(return_X_y=True)
    x_train,x_val_test,y_train,y_val_test=train_test_split(x,y,train_size=0.6)
    x_val,x_test,y_val,y_test=train_test_split(x_val_test,y_val_test,train_size=0.5)


    new=New(5,'entropy')
    new.fit(x_train,y_train)

    yp_before=new.predict(x_test)
    acc1=accuracy_score(y_test,yp_before)



    new.print_path()
    print('-------------------------')

    new.prune(x_val,y_val)
    yp_late=new.predict(x_test)

    acc2=accuracy_score(y_test,yp_late)

    new.print_path()

    print(f'acc1:{acc1:.3f}  acc2:{acc2:.3f}')

if __name__ == '__main__':
    main()




    