import numpy as np
from numpy.typing import NDArray
from numpy.linalg import norm
import matplotlib.pyplot as plt
from cluster.km.roulette import roulette_method,drawing
from sklearn.datasets import make_blobs

# from scipy.spatial.distance import cdist

def kmean(data: NDArray, k: int=2) -> tuple[NDArray,NDArray]:

    centers = roulette_method(data, k) #轮盘法直接选初始质心
    tags = np.zeros((len(data), 2))

    flag = True
    while flag:
        flag = False
        #广播方法计算每个点到最近质心的距离
        # distances = norm(data[:, None]-centers, axis=2)  #二者等价
        distances=np.sum(np.power(data[:,None]-centers,2),axis=2)
        # distances=cdist(data,centers,metric='euclidean')
        min_dis = np.min(distances, axis=1)
        min_dist_tag = np.argmin(distances, axis=1)
        if not (tags[:, 0] == min_dist_tag).all():
            flag = True
        tags[:, 0] = min_dist_tag
        tags[:, 1] = min_dis
        
        '''
        依据min_dist_tag分组 对data对数据累加和
        这一步不能保证 是否存在空的质心 分组要求k 
        但是可能经过距离计算以后 分成的组可能<=k 所以需要查找
        '''
        centers = np.zeros((k, data.shape[1]))
        np.add.at(centers, min_dist_tag, data)
        '''
        使用 bincount 统计所有 k 个簇的点数，确保长度固定为 k
        找出有质心的 和空质心 bool索引
        '''
        counts = np.bincount(min_dist_tag, minlength=k)
        valid_mask = counts > 0 
        empty_mask = counts == 0

        '''
        仅对有数据的簇计算均值 (广播 除法)
        形状(m,n)的数据 除以(m,)的数据   (m,)[:,None]-->(m,1) (None扩充维度1)
        (m,n)/(m,1)--->(m,n)/(m,n) （广播运算）
        '''
        centers[valid_mask] /= counts[valid_mask, None]
        
        # 处理空质心的  将没有质心的 地方随机选两个数据填上
        if np.any(empty_mask):
            centers[empty_mask] = data[np.random.choice(len(data), size=np.sum(empty_mask))]
            
    return centers,tags



def main():

    data, label = make_blobs(120, 2, centers=3)[:2]

    centers, tags = kmean(data, k=3)

    # drawing(data)
    # drawing(centers,color='r')

    tg=np.zeros((len(data),2))
    tg[:,0]=label
    tg[:,1]=tags[:,0]

    print(tg)
    # plt.show()

if __name__ == '__main__':
    main()



