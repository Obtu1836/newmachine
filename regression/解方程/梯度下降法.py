
import math 
fx=lambda x:x**3+math.exp(x)/2+5*x-6
'''
使用梯度下降法 并不能直接求解方程 梯度下降法 用来寻找极值。
可以对另gx=fx**2 0就是极小值点
'''

gx=lambda x:fx(x)**2

delta=1e-6
#gx的导数 用定义的表示
gx_grad=lambda x:(gx(x+delta)-gx(x))/delta

lr=0.001
x=0
while True:
    newx=x-lr*gx_grad(x)
    if abs(newx-x)<delta:
        break
    x=newx

print(newx)