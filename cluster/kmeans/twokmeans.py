import numpy as np
from numpy.typing import NDArray
from cluster.km.roulette import drawing
from cluster.km.kmeans import kmean
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

def twokmean(data:NDArray,k:int) -> tuple[NDArray,NDArray]:
    
    centers=[]#随机挑一个数据作为初始质心
    centers.append(data[np.random.choice(len(data))])

    m=data.shape[0]
    tags=np.zeros((m,2)) # tags 第一列存放标签 第二列存放距离(sse)

    while len(centers)<k:
        sse=np.inf # 初始sse设为无限大

        for i in range(len(centers)):
            '''
            遍历当前所有有质心的数据 判断哪一个簇适合分裂 判断的依据是 
            分裂后簇的sse+其他为分裂的簇的总的sse最小  
            '''
            current_data=data[tags[:,0]==i]
            split_data,split_tag=kmean(current_data,2)
            split_sse=split_tag[:,1].sum()
            non_split_sse=tags[tags[:,0]!=i,1].sum()
            total_sse=split_sse+non_split_sse

            if total_sse<sse:
                sign=i
                sse=total_sse
                sign_data=split_data
                sign_tag=split_tag
        '''
        下面两行不能调换位置
        sign_tag的值 只有0,1  如果先执行第行 假如此时适合分组的sign==1
        那么 会把sign_tag==0的部分 都换成1  再执行第一行就会把sign_tag的
        所有标签都换成len(centers) 
        '''
        sign_tag[sign_tag[:,0]==1,0]=len(centers)
        sign_tag[sign_tag[:,0]==0,0]=sign
        tags[tags[:,0]==sign,:]=sign_tag

        #下面两行可以互换
        centers[sign]=sign_data[0]
        centers.append(sign_data[1])
    
    return np.array(centers),tags

def main():
    k = 5
    data, _ = make_blobs(2500, 2, centers=k)[:2] #[:2]为了 类型检查能通过

    cents, _ = twokmean(data, k)

    drawing(data)
    drawing(cents,color='r',marker='*',s=100)
    plt.show()

if __name__ == '__main__':
    main()



