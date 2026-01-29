import math
'''
牛顿法 使用法求解方程的
         
g_f(x0)=(f(x)-f(x0))/(x-x0)  导数定义

得出 x=x0-(f(x0)/g_f(x0))  迭代公式

'''
fx=lambda x:x**3+(math.exp(x))/2+5*x-6

delta=1e-6
g_fx=lambda x:(fx(x)-fx(x-delta))/delta

x=0
while True:
    newx=x-(fx(x)/g_fx(x))
    if abs(newx-x)<delta:
        break
    x=newx

print(newx)
