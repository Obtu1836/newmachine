import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import euclidean

'''
欧氏距离 
两个向量 各个分量的差的平方的累加 最后在开方
'''


p1=np.array([1,2,3])
p2=np.array([2,3,4])

res1=euclidean(p1,p2)

res2=norm(p1-p2)

print(res1==res2)
