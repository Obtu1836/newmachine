import numpy as np
from cluster.km.twokmeans import twokmean
from sklearn.datasets import make_blobs
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def match(y_true,y_pred):
    d=max(y_pred.max(),y_true.max())+1

    w=np.zeros((d,d))

    np.add.at(w,(y_true,y_pred),1)
    '''等价于'''
    # for i in range(y_pred.size):
    #     w[y_true[i],y_pred[i]]+=1

    row_ind,col_ind=linear_sum_assignment(w.max()-w)
    
    #标签映射
    '''
    row_ind,col_ind中的数据 相同位置的元素成对应关系
    根据对应关系 通过向量化的方式将 y_pred的标签值替换为y_true中的标签值

    这个过程就像建立字典一样 
    '''
    mapping = np.zeros(w.shape[1], dtype=y_pred.dtype)
    mapping[col_ind] = row_ind # 构建字典
    y_pred[:] = mapping[y_pred]  # 通过索引(键) 取值
    
    res=accuracy_score(y_true,y_pred)


    return y_pred,res



def main():

    k=5
    data,label=make_blobs(300,2,centers=k)[:2]

    km_center,km_tags=twokmean(data,k)

    y_pred=km_tags[:,0].astype(np.int64)

    yp,txt=match(label,y_pred)

    f,(ax1,ax2)=plt.subplots(1,2,sharey=True)
    f.suptitle(f"acc:{txt:.2f}")
    ax1.scatter(data[:,0],data[:,1],c=label)
    ax1.set_title('original')
    ax2.scatter(data[:,0],data[:,1],c=yp)
    ax2.set_title(f"{twokmean.__name__}")

    plt.show()
    




if __name__ == '__main__':
    main()





