import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

class Basic(ABC):
    def __init__(self, x, y, fx, max_iters: int = 100000, lr=1e-3):

        self.fx,self.x,self.y=fx,x,y

        self.max_iters = max_iters
        self.lr = lr

        self.pos = []

    def _cal_grad(self, pos):

        delta = 1e-8
        grad = (self.fx(pos+delta)-self.fx(pos))/delta
        return grad

    @abstractmethod
    def iters_run(self):
        pass


class Normal(Basic):
    '''
    常规梯度下降： point=point-self.lr*grad
    '''

    def iters_run(self, init_pos):

        for i in range(self.max_iters):
            pos = init_pos-self.lr*self._cal_grad(init_pos)

            self.pos.append(init_pos)
            init_pos = pos
'''
动量优化算法
'''

class Momuent(Basic):
    '''
    pytorch 动量更新公式
                    theta=0
                    theta=beta*theta+grad
   梯度下降优化算法： point=point-self.lr*theta
    '''
    def __init__(self,beta,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.beta=beta
        self.theta=0

    def iters_run(self,init_pos):

        for i in range(self.max_iters):
            self.theta=self.beta*self.theta+self._cal_grad(init_pos)
            pos=init_pos-self.lr*self.theta

            self.pos.append(init_pos)
            init_pos=pos

class EMAMoment(Basic):
    """
    指数移动平均 (EMA) 动量更新
              theta=0
              theta = beta * theta + (1 - beta) * grad
    梯度下降公式:point=point-lr*theta 
    """
    def __init__(self, beta=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.theta = 0

    def iters_run(self, init_pos):
        for i in range(self.max_iters):

            grad = self._cal_grad(init_pos)
            self.theta = self.beta * self.theta + (1 - self.beta) * grad
            pos = init_pos - self.lr * self.theta

            self.pos.append(init_pos)
            init_pos=pos

'''
步长优化算法
'''
class Adagrad(Basic):

    '''
    r=0
    r=r+square(grad)
    point=point-lr/sqr(r+eps)*grad
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = 0 

    def iters_run(self, init_pos):
        for i in range(self.max_iters):
            grad=self._cal_grad(init_pos)
            self.r = self.r + np.square(grad)
            pos = init_pos - self.lr / (np.sqrt(self.r + 1e-8)) * grad
            
            self.pos.append(init_pos)
            init_pos = pos


class RMSprop(Basic):
    '''
    r=0
    r=beta*r+(1-beta)*square(grad)
    point=point-lr/sqrt(r+eps)*grad
    '''
    def __init__(self, beta=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.r = 0 
        

    def iters_run(self, init_pos):
        for i in range(self.max_iters):
            grad = self._cal_grad(init_pos)
            # 使用指数移动平均，r 不会无限增长
            self.r = self.beta * self.r + (1 - self.beta) * np.square(grad)
            pos = init_pos - self.lr / (np.sqrt(self.r + 1e-8)) * grad
            
            self.pos.append(init_pos)
            init_pos = pos

'''
动量+步长 优化算法
'''
class RMSporp_moment(Basic):

    '''
    单纯的动量与步长优化同时使用 不添加中心化的版本(不修正偏差)

           theta,r=0,0

    动量部分 theta=beta1*theta+grad
    步长部分 r=beta2*r+(1-beta2)*np.square(grad)
    梯度下降部分 point=point-lr/sqrt(r+eps)*theta
    '''

    def __init__(self, beta=0.9,beta1=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta=beta
        self.beta1 = beta1
        self.theta=0
        self.r = 0 

    def iters_run(self, init_pos):
        for i in range(self.max_iters):
            grad = self._cal_grad(init_pos)
            self.theta = self.beta * self.theta + grad
            self.r = self.beta1 * self.r + (1 - self.beta1) * np.square(grad)
            pos = init_pos - self.lr / (np.sqrt(self.r) + 1e-8) * self.theta
            
            self.pos.append(init_pos)
            init_pos = pos



class Adam(Basic):
    '''
    m = beta1 * theta+grad
    v = beta2 * v + (1 - beta2) * square(grad)
    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)
    point = point - lr * m_hat / (sqrt(v_hat) + eps)
    '''
    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def iters_run(self, init_pos):
        m = 0
        v = 0
        for i in range(1, self.max_iters + 1):
            grad = self._cal_grad(init_pos)
            
            m=self.beta1*m+(1-self.beta1)*grad  # 指数移动平均动量
            v = self.beta2 * v + (1 - self.beta2) * np.square(grad)
            
            # 偏差修正
            m_hat = m / (1 - self.beta1**i)
            v_hat = v / (1 - self.beta2**i)
            
            pos = init_pos - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
            self.pos.append(init_pos)
            init_pos = pos

class Paint:
    def __init__(self, fx, x, y):

        self.fig, self.ax = plt.subplots()
        self.fx = fx
        self.x = x
        self.y = y

    def drawing(self, *datas):

        self.ax.plot(self.x, self.y)
        self.ax.annotate('flat area', (-7, 100), (-9, 300),
                         arrowprops=dict(facecolor='green', shrink=0.05), fontsize=8)
        self.ax.annotate('local-min area', (2.5, -50), (3, -250),
                         arrowprops=dict(facecolor='orange', shrink=0.05), fontsize=8)
        self.ax.annotate('global-min area', (8.7, -1600), (5, -1250),
                         arrowprops=dict(facecolor='red', shrink=0.05), fontsize=8)
        
        scat_styles=\
        [
            {'color':'red','label':'normal'},
            {'marker':'*','label':'momentum','s':200},
            {'marker':'^','label':'adagrad','s':200},
            {'marker':'<','label':'rmsprop',"s":200},
            {'marker':'o','label':'ema',"s":100, 'alpha':0.6},
            {'marker':'>','label':'rmsp_moment',"s":100, 'alpha':0.6},

            {'marker':'X','label':'adam',"s":200}
        ]

        length=len(datas)
        scatters=[]
        for i in range(length):
            scat=self.ax.scatter([],[],**scat_styles[i])
            scatters.append(scat)

        text = self.ax.text(0.2, 0.5, '', transform=self.ax.transAxes)

        def update(n):

            for i in range(length):
                data=datas[i]
                scatters[i].set_offsets([data[n],self.fx(data[n])])

            word = f"{n}_iters"
            text.set_text(f'{word}')

            return  text,*scatters

        ani = FuncAnimation(self.fig, update, range(len(datas[0])), interval=1000,
                            repeat=False)

        self.ax.legend()
        plt.show()




def main():

    x = np.linspace(-10, 10, 1000)

    def fx(x): return 0.0001 * (x + 7)**4 * \
        (x - 1.8) * (x - 2.2) * (x - 4) * (x - 10)
    y = fx(x)

    init_pos = 5
    normal = Normal(x, y, fx, lr=0.001)
    normal.iters_run(init_pos)

    momuent = Momuent(0.95, x, y, fx, lr=1e-3)
    momuent.iters_run(init_pos)

    ema = EMAMoment(beta=0.99, x=x, y=y, fx=fx, lr=1e-3)
    ema.iters_run(init_pos)


    adagrad = Adagrad(x, y, fx, lr=1e-1)
    adagrad.iters_run(init_pos)

    rmsprop = RMSprop(beta=0.9, x=x, y=y, fx=fx, lr=1e-1)
    rmsprop.iters_run(init_pos)

    rmsp_moment=RMSporp_moment(0.9,0.9,x=x,y=y,fx=fx,lr=1e-1)
    rmsp_moment.iters_run(init_pos)

    adam = Adam(beta1=0.9, beta2=0.999, x=x, y=y, fx=fx, lr=1e-1)
    adam.iters_run(init_pos)

    normal_data, momuent_data, adagrad_data, rmsprop_data, ema_data, rmsp_moment_data, adam_data = \
        normal.pos, momuent.pos, adagrad.pos, rmsprop.pos, ema.pos, rmsp_moment.pos,adam.pos

    paint = Paint(fx, x, y)
    paint.drawing(normal_data, momuent_data, adagrad_data, rmsprop_data, 
                  ema_data, rmsp_moment_data,adam_data)


if __name__ == '__main__':
    main()
