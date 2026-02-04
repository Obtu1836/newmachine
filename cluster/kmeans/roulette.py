import numpy as np 
from numpy.typing import NDArray
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

'''
轮盘法选择质心的原理 (K-means++ 初始化):

1. 核心思想：
   让距离现有质心越远的点，有更大的概率被选为下一个质心。
   这解决了随机初始化可能导致的局部最优问题。

2. 具体步骤：
   - 计算距离：计算每个点 x 到其最近已有质心的距离 D(x)。
   - 计算概率：每个点被选为下一个质心的概率 P(x) 与 D(x)^2 成正比。
     P(x) = D(x)^2 / sum(D(x)^2)
   - 模拟轮盘：利用累加概率分布和随机数，选出下一个质心。

3. 优势：
   - 增加质心分布的多样性。
   - 兼顾随机性与确定性，提高收敛速度和聚类质量。
'''

def roulette_method(data:NDArray[np.float64],
                    k:int)->np.ndarray:
    # 随机挑选一个数据作为质心
    centers=[]
    centers.append(data[np.random.randint(len(data))])

    while len(centers)<k:
        '''
        计算 每个点到每个质心的距离平方 
        并选出每个点到该点最近的质心的距离的平方
        (采用平方 可以更分散)

        (data[:,None]-cents).shape=(m,k,n) #矩阵的广播运算
        经 sum(axis=2) shape=(m,k)  意味着 每个样本点到每个质心的距离的平方
        经 argminx(axis=1) shape=(m,) 意味着 每个样本到距离最近的质心的距离的平方 
        '''
        distance=np.sum(np.power(data[:,None]-centers,2),axis=2)
        mindistance=np.min(distance,axis=1)
        
        #将距离的平方转化为概率并进一步转化为累加概率分布
        dis_prob=mindistance/mindistance.sum()
        prob_cum=np.cumsum(dis_prob)

        '''np.searchsorted 函数 在有序数组中插入新值并保持有序
           返回插入位置的索引
           这个函数在本例中 等价于
           for i,p in dis_prob:
                if p>np.random.rand():
                     return i
        '''
        idx=np.searchsorted(prob_cum,np.random.rand())
        centers.append(data[idx])

    return np.array(centers)

def drawing(data:NDArray,**kwargs):

    plt.scatter(data[:,0],data[:,1],**kwargs)

def main():
    n=5
    data, _ = make_blobs(300, 2, centers=n)[:2]
    centers=roulette_method(data,n)
    drawing(data,color='blue',s=30)
    drawing(centers,color='red',s=80,marker='x')
    plt.show()

    
if __name__ == '__main__':
    main()




        

