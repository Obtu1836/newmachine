import torch as th
from typing import Any

from abc import ABC, abstractmethod
from torch.distributions import MultivariateNormal
from torch import Tensor

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

'''
作用:
利用 N个高斯混合 模型 通过加权 形成函数域 选定一个初始点后 观察常规梯度下降和带动量的梯度下降
的表现情况 

实现:
    1 自定义均值和协方差矩阵 生成2个 高斯分布 并可以根据自定义权重 实现2个高斯分布的叠加
        这一步 用来模拟 '地形'

    2 根据point(x,y) 计算该点的概率(z轴的值)和导数 利用pytorch中的MultivariateNormal
      和autograd 分别实现
    3 梯度下降法 根据迭代公式 w=w-lr*grad
      更新 新的点位,每一次迭代 梯度下降一次 也就会增加一个轨迹 跌代结束 也就有了下降的轨迹
    4 动画演示过程中 最快到达最低点的 并不意味着某个优化器就一定比其他的好 只能说 当前数据分布
      可能更适合该优化器 在预设的数据下 rms_moment比纯动量的还快 即使初始学习率还要小于纯动量
      这种现象的原因 更多的的可能是 在梯度较低的地形中 由于学习率自适应的原因 梯度小导致了学习率
      效果变大的情况

      关于步长自适应的总结：
      加速度不断增大的情况下 自适应学习率效果会变小 
      加速度减小的情况下 学习率效果自适应变大 
      加速度恒定时 学习率趋近于设置的学习率   
      以上状态 跟上升还是下降无关 
'''

class Base(ABC):
    def __init__(self, max_iters: int, lr: float,
                 means: Tensor, covs: Tensor, weights: Tensor):
        self.max_iters = max_iters
        self.lr = lr
        self.means: Tensor = means
        self.covs: Tensor = covs
        self.weights: Tensor = weights / weights.sum()

    def cal_xyvalue(self, points:Tensor):
        '''
        points 二维数组 表示x,y的位置 
        这个函数的功能是 根据x ,y 计算 在图像中的z 也就是概率值 
        MultivariateNormal 功能 支持多个means和covs 
        如果是points二维数组 在使用时 为了实现广播 points 需要主动扩展维度:,None,:
        如果是points是1维数组 可以直接使用
        '''

        dist = MultivariateNormal(
            self.means, self.covs).log_prob(points[:, None, :])#传入的是数组 需要主动扩展维度
        res = th.exp(dist)
        res = self.weights*res

        return res.sum(dim=1)

    def make_data(self, xlimits: tuple[int, ...], ylimits: tuple[int, ...])\
                                         -> tuple[Tensor, Tensor, Tensor]:
        '''
        生成数据 作为'地形'
        '''

        x = th.linspace(*xlimits)  #(m,)
        y = th.linspace(*ylimits)  #(m,)
        x_l, y_l = xlimits[-1], ylimits[-1] # m m
        xx, yy = th.meshgrid(x, y, indexing='ij')#(m,m) (m,m)
        xy = th.dstack([xx, yy]).reshape(-1, 2)#(m,m,2)-->(m*m,2)
        zz = self.cal_xyvalue(xy).reshape(x_l, y_l)#(m*m,)-->(m,m)
        return xx, yy, zz

    def _cal_point_grad(self, point: Tensor) -> Tensor:
        '''
        计算每个点的导数

        th.autograd.grad 返回的元素 第一个是单纯的Tensor 不带有任何跟梯度相关的东西
        独立于反向传播 所以可以直接拿来运算
        '''
        point=point.detach().clone()#剥离计算图 防止累加
        point.requires_grad = True # 计算梯度 必须设置为True

        dist = MultivariateNormal(self.means, self.covs)
        log_p = dist.log_prob(point)
        loss = (th.exp(log_p)*self.weights).sum()
        grad = th.autograd.grad(loss, point)[0] # 利用autograd 简单写法

        return grad

    @abstractmethod
    def grad_dense(self, point):
        ...

class MulGaussNormal(Base):
    '''常规模式下的梯度下降法'''
    def grad_dense(self, point:Tensor):
        '''
        通过传入一个初始的点位 计算当前点位的梯度 在迭代 跟踪新的点位 形成轨迹
        虽然传入的point 只是单纯的Tensor 但是 将point 传入_cal_point_grad以后
        这个函数内部会对point进行一些处理 导致point的requires_grad=True(可变对象经函数后会变化)
        使用th.no_grad 可以保证 point=point-self.lr*grad 不会参与到自动求导 确保只是进行数值更新
        '''

        trajectory = []
        for i in range(self.max_iters):
            grad = self._cal_point_grad(point)
            with th.no_grad():
                point = point-self.lr*grad
            trajectory.append(point)

        return trajectory


class MulGaussMoment(Base):
    '''带有动量的梯度下降法'''

    def __init__(self, beta, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.beta = beta

    def grad_dense(self, point):

        trajectory = []
        theta = 0
        for i in range(self.max_iters):
            grad = self._cal_point_grad(point)
            with th.no_grad():
                #带动量的梯度更新公式
                theta = self.beta*theta+grad
                point = point-self.lr*theta
            trajectory.append(point)

        return trajectory
    
class RMSprop(Base):
    '''
    r=0
    r=beta*r+(1-beta)*square(grad)
    point=point-lr/sqrt(r+eps)*grad
    '''
    def __init__(self, beta=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.r = 0 
        

    def grad_dense(self, init_pos):
        trajectory = []
        for i in range(self.max_iters):
            grad = self._cal_point_grad(init_pos)
            # 使用指数移动平均，r 不会无限增长
            self.r = self.beta * self.r + (1 - self.beta) * th.square(grad)
            pos = init_pos - self.lr / (th.sqrt(self.r + 1e-8)) * grad
            
            trajectory.append(init_pos)
            init_pos = pos

        return trajectory

class RMSporp_moment(Base):

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

    def grad_dense(self, init_pos):
        trajectory = []
        for i in range(self.max_iters):
            grad = self._cal_point_grad(init_pos)
            self.theta = self.beta * self.theta + grad
            self.r = self.beta1 * self.r + (1 - self.beta1) * th.square(grad)
            pos = init_pos - self.lr / (th.sqrt(self.r) + 1e-8) * self.theta
            
            trajectory.append(init_pos)
            init_pos = pos
        return trajectory
    
class Adam(Base):
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

    def grad_dense(self, init_pos):
        m = 0
        v = 0
        trajectory=[]
        for i in range(1, self.max_iters + 1):
            grad = self._cal_point_grad(init_pos)
            
            m=self.beta1*m+(1-self.beta1)*grad  # 指数移动平均动量
            v = self.beta2 * v + (1 - self.beta2) * th.square(grad)
            
            # 偏差修正
            m_hat = m / (1 - self.beta1**i)
            v_hat = v / (1 - self.beta2**i)
            
            pos = init_pos - self.lr * m_hat / (th.sqrt(v_hat) + self.eps)
            
            trajectory.append(init_pos)
            init_pos = pos
        return trajectory



class Paint:
    def __init__(self, *args):

        self.x, self.y, self.z = args
        self.fig = plt.figure(figsize=(10,8),dpi=80)
        self.ax: Any = self.fig.add_subplot(projection='3d')

    def draw_surf(self):

        self.ax.plot_surface(self.x, self.y, self.z, alpha=0.8)
        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Y Axis')
        self.ax.set_zlabel('Z Axis')

    def draw_scatter_insted(self, points):

        st_style = [
            {"marker": "^", "color": "red", "s": 200},
            {"marker": "*", "color": "blue", "s": 200},
            {'marker':'<','color':'orange','s':200},
            {'marker':'>','color':'pink','s':200},
            {'marker':'X','color':'grey','s':200},

            ]

        tx_style = [
            [0.1, 0.9],
            [0.9, 0.9],
            [0.05,0.1],
            [0.95,0.1],
            [0.5,0.1]]

        scatters = []
        data_list = []
        texts = []

        label = self.ax.text2D(0.5, 0.95, '', transform=self.ax.transAxes)
        for i, point in enumerate(points):
            st_st = st_style[i]
            scatter = self.ax.scatter3D([], [], [], **st_st)
            tx = tx_style[i]
            txt = self.ax.text2D(*tx, s='', transform=self.ax.transAxes)

            scatters.append(scatter)
            data_list.append(point.detach().numpy())
            texts.append(txt)

        def update(n):

            for i, scatter in enumerate(scatters):
                x, y, z = data_list[i][n]
                scatter._offsets3d = ([x], [y], [z+0.001])
                marker = st_style[i]['marker']
                texts[i].set_text(
                    f'{marker} \nPos: {x:.2f}, {y:.2f}\nZ: {z:.4f}')
                label.set_text(f'{n}-iters')
            return scatter, label

        self.ani = FuncAnimation(self.fig, update, frames=range(
            len(data_list[0])), interval=100)
        plt.show()

    def run(self, points):

        self.draw_surf()
        self.draw_scatter_insted(points)


def cal_xyz(base: Base, point: list[float] | Tensor) -> Tensor:
    '''
    将所有记录到轨迹 包含x,y的一系列坐标 通过cal_xyvalue 计算出z 
    然后拼接到一起 返回一个2维数组  m个轨迹 每个轨迹x,y,z三个坐标
    '''
    trajectory_list = base.grad_dense(point)#记录踪迹[tensor1,tensor2]
    trajectory: Tensor = th.stack(trajectory_list)#合并
    z: Tensor = base.cal_xyvalue(trajectory)
    xyz: Tensor = th.column_stack([trajectory, z[:, None]])#区别于numpy 这个需要扩展维度
    return xyz


def main():

    xlims = (-3, 3, 120)
    ylims = (-3, 3, 120)

    means = th.tensor([[-1.5, -1.5], [1.5, 1.5]])
    covs = th.tensor([[[1.5, 1], [1, 1.5]],
                      [[1, 0.5], [0.5, 1]]])
    weights = th.tensor([1.2, 1.5])
    '''
    xlims,ylims
    means,covs,weights这些自定义参数 可以自由设置 
    支持添加更多个高斯分布 只要保持 长度一致即可 协方差矩阵符合即可
    '''

    # means=th.tensor([[0.2,0.5],[1.2,1.8],[-2.3,1.3]])
    # covs=th.tensor([[[2,0.3],[0.3,2]],
    #                 [[1.5,0.6],[0.6,1.5]],
    #                 [[2.5,1.2],[1.2,2.5]]])

    # weights=th.tensor([0.3,0.4,0.5])

    lr = 0.3
    max_iters = 1000
    mulgauss = MulGaussNormal(max_iters, lr, means, covs, weights)
    xs, ys, zs = mulgauss.make_data(xlims, ylims)

    mulmoment = MulGaussMoment(0.9, max_iters, lr, means, covs, weights)
    rmsprop=RMSprop(0.9,max_iters,0.01,means,covs,weights)
    rms_moment=RMSporp_moment(0.9,0.9,max_iters,0.01,means,covs,weights)
    adam=Adam(0.9,0.9,1e-8,max_iters,0.01,means,covs,weights)

    # point=th.tensor([0.7,0.7],dtype=th.float32)
    point = th.tensor([0.8,0.8],dtype=th.float32)

    normalxyz = cal_xyz(mulgauss, point)
    momentxyz = cal_xyz(mulmoment, point)
    rmspropxyz=cal_xyz(rmsprop,point)
    rms_momentxyz=cal_xyz(rms_moment,point)
    adamxyz=cal_xyz(adam,point)

    paint = Paint(xs, ys, zs)
    paint.run([normalxyz, momentxyz,rmspropxyz,rms_momentxyz,adamxyz])


if __name__ == '__main__':
    main()
