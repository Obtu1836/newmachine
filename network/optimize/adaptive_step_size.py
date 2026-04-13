import numpy as np
import matplotlib.pyplot as plt

def calculte_step_length(x,grad,warmtimes):
    '''
    自适应步长 更新公式  前一段是进行warmtimes的预热 防止前几次因为
    梯度累加  影响判断
    '''
    v0=0
    beta=0.9
    for i in range(warmtimes):
        v0=beta*v0+(1-beta)*grad[i]**2
    
    rs=[] #根据EMA 进行更新
    for i in range(warmtimes,len(x)):
        v0=beta*v0+(1-beta)*grad[i]**2
        r=1/np.sqrt(v0) #因为只查看 自适应步长的变化 所以不乘当前梯度和学习率了
        rs.append(r)
    
    return rs


class Paint:
    def __init__(self,xstart,xend,name:str):

        self.xstart=xstart
        self.xend=xend

        self.fig,self.ax=plt.subplots(1,3,figsize=(10,8),dpi=100)
        self.fig.suptitle(name)
    
    def paint(self,x,y,y1,x2,y2):

        self.ax[0].plot(x,y)
        self.ax[0].set_title('function_value')
        self.ax[1].plot(x,y1)
        self.ax[1].set_title('|function_grad|')
        self.ax[2].plot(np.arange(len(x2)),y2)
        self.ax[2].set_title('adaptive_step_size')
        


def concave_function(x):
    # 凹函数 返回 x中每个点的函数值和导数 
    return np.log(-x),np.abs(-1/x)

def convex_function(x):
    #凸函数 返回每个x点的函数值和导数
    return np.exp(-x),np.abs(-np.exp(-x))
'''
因为 步长更新公式中 要么是梯度的平方 要么是开方 所以在返回导数时 添加了abs
这个导数 只反应陡峭程度  跟符号没关系 
'''

def const_function(x):
    '''常量函数  导数g可以为任意常量 本例中设为1
    从公式中 推导得出 当梯度维持恒定时 步长系数趋向于1/g
    (lr*1/g*g  -->lr  当导数恒定时 每次更新都是lr )
    '''
    return x,np.full(len(x),1)

def create_function(sign:int):
    dict={1:convex_function,2:concave_function,3:const_function}
    return dict[sign]

def main():

    length=50
    start,end=-5,-0.01
    x=np.linspace(start,end,length)
    
    function=create_function(1)
    name=function.__name__
    y,g=function(x) #每个点的函数值和导数

    warmtimes=20 # 启动预热20个迭代次数 使其充分预热
    rs=calculte_step_length(x,g,warmtimes)
    paint=Paint(start,end,name)
    paint.paint(x,y,g,x[warmtimes:],rs)

    plt.show()

if __name__ == '__main__':
    main()
