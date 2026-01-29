from scipy.spatial.distance import cityblock
import numpy as np

'''
曼哈顿距离
两个向量 中各个分量的差的绝对值的累加
'''

p1=np.array([1,2,3])
p2=np.array([2,3,4])

res=cityblock(p1,p2)

rek=np.abs(p1-p2).sum()

print(res==rek)











