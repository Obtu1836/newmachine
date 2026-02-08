import numpy as np 
from numpy.linalg import solve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Lwlr:
    def __init__(self,sigma:float=0.2):

        self.sigma=sigma
    
    def fit(self,trainx,trainy):

        self.x=np.column_stack([trainx,np.ones(len(trainx))])
        self.y=trainy[:,None]

        return self

    def predict(self,test):
        test=np.column_stack([test,np.ones(len(test))]) #(m,2)
        dis2=np.sum(np.power(test[:,None]-self.x,2),axis=2) #(m,n)
        dis2=np.exp(-dis2/(2*self.sigma**2)) #(m,n)
        
        # bx=np.stack([self.x]*len(test),axis=0) #(m,n,r)
        # by=np.stack([self.y]*len(test),axis=0) #(m,n,1)
        bx=self.x[None,...]
        by=self.y[None,...]        
        bxT=bx.transpose(0,2,1)  #(m,r,n)
        bxTq=bxT*dis2[:,None,:] # #(m,r,n)
        bxTqx=bxTq@bx  #(m,r,r)

        reg=np.eye(bxTqx.shape[-1])*1e-5
        bxtqy=bxTq@by  # (m,r,1)
        bw=solve(bxTqx+reg,bxtqy) #(m,r,r) (m,r,1) -->(m,r,1)

        yp=test[:,None,:]@bw  #(m,1,r)@(m,r,1) -->(m,1)
        return yp.ravel()
    
class Paint:

    def __init__(self,x,y):

        self.x=x
        self.y=y

        self.fig,self.ax=plt.subplots(figsize=(10,6))
        self.ax.scatter(x,y,color='g')
        self.line,=self.ax.plot([],[],lw=2,color='r')
        self.text=self.ax.text(0.1,0.9,'',transform=self.ax.transAxes)

    def drawing(self,test):

        sigmas=[2,1,0.5,0.3,0.1,0.05]

        def update(n):
            lwlr=Lwlr(sigmas[n])
            lwlr.fit(self.x,self.y)
            yp=lwlr.predict(test)
            self.line.set_data(test,yp)
            self.text.set_text(f'sigma: {sigmas[n]}')
            return self.line,self.text
        
        self.ax.set_xlim(test.min()-1,test.max()+1)

        ani=FuncAnimation(self.fig,update,range(len(sigmas)),
                          interval=2000,repeat=False)
        plt.show()

def main():
    x=np.linspace(-3,3,20)
    y=np.sin(x)+np.random.randn(len(x))*0.3
    test=np.linspace(-3,3)

    
    paint=Paint(x,y)
    paint.drawing(test)


if __name__ == '__main__':
    main()



