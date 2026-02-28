import numpy as np 
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt 

def make_data(xlimits,ylimits,mean=[0,0],cov=[[1,0],[0,1]]):
    x=np.linspace(*xlimits)
    y=np.linspace(*ylimits)
    x_length=xlimits[-1]
    ylength=ylimits[-1]
    xx,yy=np.meshgrid(x,y)
    
    xy=np.dstack([xx,yy]).reshape(-1,2)

    p=multivariate_normal(mean,cov).pdf(xy)
    zz=p.reshape(x_length,ylength)
    
    return xx,yy,zz

class Paint:
    def __init__(self,x,y,z):

        self.fig=plt.figure()
        self.ax=self.fig.add_subplot(111,projection='3d')
        self.ax.plot_surface(x,y,z,cmap='viridis')

        plt.show()


def main():
    xlimits=(-5,5,120)
    ylimits=(-3,3,120)

    x, y, z = make_data(xlimits, ylimits)

    # 计算 x 方向的跨度作为偏移量，使拼接后的图形在空间上连续
    x_range = xlimits[1] - xlimits[0] 

    # 沿着 axis=1 (x 方向) 拼接
    xc = np.concatenate([x, x + x_range], axis=1)
    yc = np.concatenate([y, y], axis=1)
    zc = np.concatenate([z, -z], axis=1)

    # print(f"原始形状: {x.shape}")
    # print(f"拼接后形状: {xc.shape}")

    # 使用 Paint 类进行绘制

    # paint = Paint(xc, yc, zc)

if __name__ == '__main__':
    main()

