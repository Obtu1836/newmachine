from jax import grad
from math import factorial
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
from typing import Callable

# 泰勒展开式: f(x) ≈ f(a) + f'(a)(x-a) + f''(a)/2! * (x-a)^2 + ...
'''
对一个任意的函数在a点附近 用一个多项式逼近 两个函数图像 距离a点越近 表现越相似
'''


def ori_fun(x):
    return jnp.sin(x)


def taylor(fun: Callable, # ori_fun 原函数
           num: int,   #函数点位
           length: int, #项数
           x: float):  # num附近的x
    '''
    
    模拟taylor的展开式的项数 length=项数
    '''

    res = [fun(num)]    # f(a)
    for i in range(1, length):
        fun = grad(fun)  #通过迭代 计算 每项的导数
        k = fun(float(num))/factorial(i)*(x-num)**i # fI(num)(x-num)/i!
        res.append(k)
    return sum(res)


def main():

    num, length = 5, 3  #函数的点位·· ·  泰勒展开项数
    part_taylor = partial(taylor, ori_fun, num, length)

    xs=jnp.linspace(num-1,num+1)  # 图像区间 x轴 
    y1=[ori_fun(x) for x in xs]  # 原函数的 y 
    y2=[part_taylor(x) for x in xs] # taylor的y

    f,ax=plt.subplots(1,2)
    f.set_label(f'{num} img')
    ax[0].plot(xs,y1,color='r')
    ax[0].set_title('ori_fun')
    ax[1].plot(xs,y2,color='g')
    ax[1].set_title('taylor')
    plt.show()


if __name__ == '__main__':
    main()
