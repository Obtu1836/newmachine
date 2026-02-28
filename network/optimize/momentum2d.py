import numpy as np
from collections import deque
from abc import ABC,abstractmethod
import matplotlib.pyplot as plt
from pathlib import Path

from matplotlib.animation import FuncAnimation,PillowWriter


class Basic(ABC):
    def __init__(self,x,y,fx,max_iters:int=100000,lr=1e-3):
        
        self.fx=fx
        self.x=x
        self.y=y

        self.max_iters=max_iters
        self.lr=lr

        self.pos=[]
        self.deque=deque(maxlen=10)
    
    def _cal_grad(self,pos):
        delta=1e-8
        grad=(self.fx(pos+delta)-self.fx(pos))/delta
        return grad
    
    @abstractmethod
    def iters_run(self):
        pass

class Normal(Basic):

    def iters_run(self,init_pos):

        for i in range(self.max_iters):
            pos=init_pos-self.lr*self._cal_grad(init_pos)
            self.deque.append(pos)

            if len(self.deque)>8:
                if np.allclose(np.array(self.deque).mean(),pos,rtol=1e-3):
                    break
            self.pos.append(init_pos)

            init_pos=pos

class Momuent(Basic):
    def __init__(self,beta,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.beta=beta
        self.theta=0

    def iters_run(self,init_pos):

        for i in range(self.max_iters):
            self.theta=self.beta*self.theta-self.lr*self._cal_grad(init_pos)
            pos=init_pos+self.theta
            self.deque.append(pos)

            if len(self.deque)>8:
                if np.allclose(np.array(self.deque).mean(),pos,rtol=1e-3):
                    break

            self.pos.append(init_pos)
            init_pos=pos


class Paint:
    def __init__(self,fx,x,y):

        self.fig,self.ax=plt.subplots()
        self.fx=fx
        self.x=x
        self.y=y
    
    def drawing(self,data1,data2):

        self.ax.plot(self.x,self.y)
        self.ax.annotate('flat area',(-7,100),(-9,300),
                         arrowprops=dict(facecolor='green', shrink=0.05), fontsize=8)
        self.ax.annotate('local-min area',(2.5,-50),(3,-250),
                         arrowprops=dict(facecolor='orange', shrink=0.05), fontsize=8)
        self.ax.annotate('global-min area',(8.7,-1600),(5,-1250),
                         arrowprops=dict(facecolor='red', shrink=0.05), fontsize=8)
        
        scatter_normal=self.ax.scatter([],[],color='red',label='normal')
        scatter_momuent=self.ax.scatter([],[],color='blue',marker='*',s=200,label='momuent')
        text=self.ax.text(0.2,0.5,'',transform=self.ax.transAxes)

        length=len(data1)

        def update(n):

            scatter_normal.set_offsets([data1[n],self.fx(data1[n])])
            scatter_momuent.set_offsets([data2[n],self.fx(data2[n])])

            word=f"{n}_iters"
            if n==length-1:
                word='stop'
            
            text.set_text(f'{word}')

            return scatter_normal,scatter_momuent,text

        ani=FuncAnimation(self.fig,update,range(length),interval=200,
                          repeat=False)
        
        self.ax.legend()
        path=Path(__file__).resolve().parent
        ani.save(path/ 'new1.gif',writer=PillowWriter(fps=18))
        plt.show()

def align(data1, data2):
    max_len = max(len(data1), len(data2))
    
    # np.pad 的 (0, n) 表示：起始端填0个，末尾端填n个
    # mode='edge' 表示：使用边缘值（即最后一个元素）进行填充
    res1 = np.pad(data1, (0, max_len - len(data1)), mode='edge')
    res2 = np.pad(data2, (0, max_len - len(data2)), mode='edge')
    
    return res1, res2


def main():

    x=np.linspace(-10,10,1000)
    fx=lambda x:0.0001 * (x + 7)**4 * (x - 1.8) * (x - 2.2) * (x - 4) * (x - 10)
    y=fx(x)

    init_pos=-1.5
    normal=Normal(x,y,fx,lr=1e-3)
    normal.iters_run(init_pos)

    momuent=Momuent(0.95,x,y,fx,lr=1e-3)
    momuent.iters_run(init_pos)

    normal_data,momuent_data=normal.pos,momuent.pos
    normal_data,momuent_data=align(normal_data,momuent_data)
    
    paint=Paint(fx,x,y)
    paint.drawing(normal_data,momuent_data)


if __name__ == '__main__':
    main()



        




