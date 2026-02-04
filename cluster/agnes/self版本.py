import numpy as np 
from scipy.spatial.distance import cdist

from sklearn.datasets import make_moons,make_circles
import matplotlib.pyplot as plt

'''
单链层次聚类 
    1 初始将每个样本看作一个簇 
    2 计算每个簇到其余簇的距离 (常用欧式距离)        ---- 循环
    3 选择最近的两个簇 然后合并成一个簇              ---
    4 直到 只剩下k个簇

    算法 需要计算大量的距离 循环以及np.r_,且涉及列表和np数组的
    转化 耗费大量时间  但是 易于理解
'''

def opts(data,k):

    while len(data)>k:
        
        sign=(-1,-1) #记录哪两个簇 合并
        dis=np.inf  #设置初始比较的 
        
        for i in range(len(data)-1):
            d1=np.atleast_2d(data[i]) # 确定当前簇  保证至少是2维的 因为cdist需要输入二维数组
            
            for j in range(i+1,len(data)):# 遍历其余的簇
                d2=np.atleast_2d(data[j]) # 确定其余的簇的当前簇
                jdis=cdist(d1,d2).min() # 计算当前簇的所有点和其余簇的当前簇所有距离 并找到最小值的点的距离
                if jdis<dis: 
                    dis=jdis #通过比较 最小距离 确定出 需要合并的簇
                    sign=(i,j) # 记录哪两个簇需要合并
        a,b=sign
        data[a]=np.r_[np.atleast_2d(data[a]),np.atleast_2d(data[b])]# 合并
        data.pop(b) # 删除被合并的簇

    return data

def main():
    
    # x,y=make_moons(100,noise=0.05)
    x,y=make_circles(200,factor=0.5)

    xt=x.tolist()
    xx=[np.array(x) for x in xt]

    res=opts(xx,2)

    fig,ax=plt.subplots(1,2,figsize=(10,6))

    ax[0].scatter(x[:,0],x[:,1],c=y)
    a,b=res
    ax[1].scatter(a[:,0],a[:,1],color='r')
    ax[1].scatter(b[:,0],b[:,1],color='g')
    plt.show()

if __name__ == '__main__':
    main()




           