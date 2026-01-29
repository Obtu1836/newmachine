import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cosine

'''
余弦相似度
cos(θ) = (A · B) / (||A|| * ||B||)
A,B为向量 
A*B 为 A中的各个分量与B中的各个分量点乘 在累加
'''

u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

res = cosine(u, v) # 计算的是余弦距离  所以最终的相似度需要1-余弦距离

rek =u[None,:].dot(v[:,None])/(norm(u)*norm(v))

print(rek.ravel()[0]==1-res)
