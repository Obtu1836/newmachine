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
                theta = self.beta*theta-self.lr*grad
                point = point+theta
            trajectory.append(point)

        return trajectory


class Paint:
    def __init__(self, *args):

        self.x, self.y, self.z = args
        self.fig = plt.figure()
        self.ax: Any = self.fig.add_subplot(projection='3d')

    def draw_surf(self):

        self.ax.plot_surface(self.x, self.y, self.z, alpha=0.8)
        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Y Axis')
        self.ax.set_zlabel('Z Axis')

    def draw_scatter_insted(self, points):

        st_style = [
            {"marker": "^", "color": "red", "s": 200},
            {"marker": "*", "color": "blue", "s": 200}]

        tx_style = [
            [0.1, 0.9],
            [0.9, 0.9]]

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

    lr = 0.3
    max_iters = 1000
    weights = th.tensor([1.2, 1.5])
    mulgauss = MulGaussNormal(max_iters, lr, means, covs, weights)
    xs, ys, zs = mulgauss.make_data(xlims, ylims)

    mulmoment = MulGaussMoment(0.9, max_iters, lr, means, covs, weights)

    # point=[0.7,0.7]
    point = th.tensor([1.2, 1.2])

    normalxyz = cal_xyz(mulgauss, point)
    momentxyz = cal_xyz(mulmoment, point)

    paint = Paint(xs, ys, zs)
    paint.run([normalxyz, momentxyz])


if __name__ == '__main__':
    main()
