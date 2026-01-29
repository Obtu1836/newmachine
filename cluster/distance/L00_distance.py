from scipy.spatial.distance import chebyshev
import numpy  as np

'''
切比雪夫距离
两个向量间 各个分量间的差的最大值（绝对值）
'''
p1=np.array([1,2,-3])
p2=np.array([2,5,10])


res=chebyshev(p1,p2)
rek=np.abs(p1-p2).max()
print(res==rek)
