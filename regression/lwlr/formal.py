import sys
import numpy as np
from numpy.typing import NDArray
from numpy.linalg import lstsq

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

'''
lwlr不生成具体的模型 将每条测试数据直接与训练数据直接计算

计算过程： 计算每条测试数据与训练数据的dis 将dis 通过高斯核转为权重
在通过加权的最小二乘法 解出w  再将测试数据与w 矩阵乘法 得出testy
'''

class New:
    def __init__(self,trainx:NDArray,trainy:NDArray):

        self.trainx=trainx
        self.trainy=trainy

        self.fig,self.ax=plt.subplots(figsize=(10,6))
        self.line,=self.ax.plot([],[],color='r',alpha=0.8)
        self.text=self.ax.text(0.1,0.9,'',transform=self.ax.transAxes,
                               fontsize=20)

    def cal_testdata(self,tests:NDArray,k:int)->NDArray:
        
        x=np.c_[self.trainx,np.ones_like(self.trainx)]
        y=self.trainy[:,None]

        '''通过广播算法 计算出所有的测试数据与训练数据的dis 通过高斯核转成权重'''
        all_dis=np.sum(np.power(tests[:,None]-x,2),axis=2) 
        all_weights=np.exp(-all_dis/(2*k**2))

        res=[]
        for i in range(len(tests)):
            
            qs=x.T*all_weights[i] #这一步代替了x.T.dot(np.diag(all-weights[i])) 因为创建diag耗时
            w,_,_,_=lstsq(qs.dot(x),qs.dot(y))
            p=(tests[i][None,:].dot(w)).ravel()
            res.append(p)
        return np.array(res)
            

    def predict(self,tests:NDArray):
        '''
        通过闭包的方式设计结构 因为update函数接受一个参数 而update内部cal_testdata
        需要两个参数 所以testall这个参数 由外函数传入  通过闭包的方式 实现了参数分离的效果
        '''
        testall=np.c_[tests,np.ones_like(tests)]
    
        def update(k):

            output=self.cal_testdata(testall,k)
            self.line.set_data(tests,output)
            self.text.set_text(f'gauss_ker: {k:.3f}')

            return self.line,self.text
        
        self.ax.scatter(self.trainx,self.trainy,)
        self.ax.set_xlim(self.trainx[0]-1,self.trainx[-1]+1)
        self.ani=FuncAnimation(self.fig,update,frames=np.linspace(1,0.1,10),
                          blit=True,interval=1000,repeat=False)
        plt.show()

def main():

    trainx=np.linspace(-3,3,30)
    trainy=np.sin(trainx)+np.random.randn(len(trainx))*0.2

    tests=np.linspace(-3,3)

    cls=New(trainx,trainy)

    cls.predict(tests)

if __name__ == '__main__':
    main()

    

