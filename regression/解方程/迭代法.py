
import math
fx=lambda x:x**3+math.exp(x)+5*x-6
'''
求解fx=0时  x的值
'''
x=0
delta=1e-5
while True:
    newx=(6-(x**3+math.exp(x)/2))/5
    if abs(newx-x)<delta:
        break
    x=newx

print(newx)
