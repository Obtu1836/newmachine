import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
'''
袋装集成  一般子分类器都为同一种 常用的有随机森林
以二分类为例
设 基分类器 准确分类的概率为p  集成分类器中 共有n个基分类器  
假设 集成分类器中 有k个为正确分类 (k>=n//2) 即 有超过一半的基分类器正确预测
则 集成分类器的预测成功的概率为
comb(n,k)*p**k*(1-p)**(n-k)+
comb(n,k+1)*p**k+1*(1-p)**(n-(k+1))+....
comb(n,k+2)*p**(k+2)*(1-p)**(n-(k+2))
'''

def fun(n,p):
    total=0
    for k in range(n//2+1,n+1):
        total+=comb(n,k)*p**k*(1-p)**(n-k)
    return total

ps=np.linspace(0,1)

ns=[5,10,12,20,35]

for n in ns:
    ys=[]
    for p in ps:
        y=fun(n,p)
        ys.append(y)
    plt.plot(ps,ys,label=f'nums={n}')
plt.legend()
plt.show()

    





