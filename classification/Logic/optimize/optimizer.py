import numpy as np 
from numpy.typing import NDArray


from typing import Literal


ALL_OPTMIZER={}

def auto_add(cls):
    ALL_OPTMIZER[cls.__name__]=cls
    return cls

@auto_add
class SGDMomentum:
    def __init__(self,initv:NDArray,beta1:float=0.9):

        self.initv=initv
        self.beta1=beta1

    def step(self,grad):
        self.initv=self.initv*self.beta1+grad
        return self.initv,None

@auto_add
class EMAMomentum:
    def __init__(self,initv:NDArray,beta1:float=0.9):
        
        self.initv=initv
        self.beta1=beta1

        self.count=1
    
    def step(self,grad):
        self.initv=self.initv*self.beta1+(1-self.beta1)*grad
        v_hat=self.initv/(1-self.beta1**self.count)
        self.count+=1
        return v_hat,None

@auto_add
class AdaGrad:
    def __init__(self,initr:NDArray):
        
        self.initr=initr
    
    def step(self,grad):
        self.initr=self.initr+np.square(grad)
        return self.initr,None

@auto_add
class RMSProp:
    def __init__(self,initr:NDArray,
                 initv:NDArray,
                 beta1:float=0.9,
                 beta2:float=0.9,
                 ):
        
        self.initr=initr
        self.initv=initv
        self.beta1=beta1
        self.beta2=beta2
    
    def step(self,grad:NDArray):
        self.initr=self.beta1*self.initr+(1-self.beta1)*np.square(grad)
        self.initv=self.beta2*self.initv+(1-self.beta2)*grad
        return self.initv,self.initr

@auto_add
class Adam:
    def __init__(self,initv:NDArray,
                      initr:NDArray,
                      beta1:float=0.9,
                      beta2:float=0.9,
                  ):
        self.initv=initv
        self.initr=initr
        self.beta1=beta1
        self.beta2=beta2

        self.count=1
    
    def step(self,grad):

        self.initv=self.beta1*self.initv+(1-self.beta1)*grad
        self.initr=self.beta2*self.initr+(1-self.beta2)*np.square(grad)

        v_hat=self.initv/(1-self.beta1**self.count)
        r_hat=self.initr/(1-self.beta2**self.count)

        self.count+=1

        return v_hat,r_hat
    
    
def create_optimizer(name:str,**kwargs):
    cls=ALL_OPTMIZER.get(name)
    if not cls:
        raise KeyError('输入当优化器名字错误')
    import inspect
    required_parmas=inspect.signature(cls.__init__).parameters
    # print(required_parmas)

    required={k:v for k,v in kwargs.items() if k in required_parmas}
    # for req in required:
    #     if req not in kwargs:
    #         raise ValueError(f'缺失必要的参数 {req}')
    return cls(**required)

if __name__ == '__main__':
    
    optim=create_optimizer('SGDMomentum',initv=np.zeros((5,1)),initr=np.zeros((5,1)),
                     beta1=0.9)
    
    print(optim.initv,optim.beta1)
    



        

        
