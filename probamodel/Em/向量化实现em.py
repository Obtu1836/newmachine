import numpy as np 

'''
共有3处 使用了向量化操作 
1 计算 data的每行唯一值的数量 等同于 df.apply(lambda x:x.value_counts(),axis=1)
2 np.power([1-p,p],nums) 计算似然函数
3 2个2维矩阵 遍历互乘列 操作
'''

class Em:
    def __init__(self,pab):

        self.pab=pab

    def fit(self,data):

        # 统计 每行X Q的数量
        uni=np.unique(data)
        self.counts=(data[...,None]==uni).sum(axis=1)

        self.p_num=len(self.pab) # 记录概率数量
        self.d_num=len(data) # 记录样本条数

        while True:
            #循环 迭代 em算法

            looklike=self.e_step()
            pab=self.m_step(looklike)
            if np.allclose(self.pab,pab):
                break
            self.pab=pab
        
        
    def e_step(self,):

        neg_pab=1-self.pab
        complete_pab=np.array([neg_pab,self.pab]).T
        looklike=np.power(complete_pab[:,None],self.counts).prod(axis=2)
        looklike=looklike.T
        looklike=looklike/looklike.sum(axis=1,keepdims=True)#似然函计算期望
        return looklike
    
    def m_step(self,z):
        
        # 利用广播算法 计算交叉列乘 
        rp=(z[...,None]*self.counts[:,None,:]).sum(axis=0)
        pab=rp[:,-1]/rp.sum(axis=1)
        return pab
        
    
def main():

     data = np.array([
        [0, 1, 1, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 1, 0, 0, 0, 0, 0, 1]
    ])
     
     model=Em(np.array([0.6,0.5]))

     model.fit(data)
     print(model.pab)
    
if __name__ == '__main__':
    main()
