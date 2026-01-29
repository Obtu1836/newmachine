import numpy as np 
from sklearn.metrics import log_loss

'''
手动实现 交叉熵  

交叉熵 由 概率 和标签 构成 其中概率 二维数组的形状为 (m,k) m个样本k个类别 k>=2  且经过了softmax
                           标签 一维数组 形状(m,)  m个样本的标签
'''

def softmax(p):
    
    exp=np.exp(p)
    return exp/exp.sum(axis=1,keepdims=True)

def sk(p,y):
    return log_loss(y,p)

def self(p,y):

    k=len(np.unique(y))
    mask=np.eye(k)[y]
    return -(mask*np.log(p)).sum()/len(y)

def main():
    p=np.random.rand(10,3)
    y=np.array([0,1,2,0,2,2,1,2,0,1])
    softmaxp=softmax(p)

    result_sk=sk(softmaxp,y)
    result_self=self(softmaxp,y)

    print(f"{'result_sk:':<15} {result_sk:.3f}")
    print(f"{'result_self:':<15} {result_self:.3f}")



if __name__ == '__main__':
    main()




