import sympy
'''
一元微分 是向量  二元微分是矩阵(绝大多数是对称矩阵) 3元以上是张量

'''

x,y=sympy.symbols('x y')

f=x**2+x*y+y**3

print(sympy.hessian(f,(x,y)))