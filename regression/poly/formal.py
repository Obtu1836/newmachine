import numpy as np 
import  matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class Poly:
    def __init__(self):

        self.trainx=np.linspace(-2,2,20)
        self.trainy=np.sin(self.trainx)+np.random.rand(len(self.trainx))*0.3
        self.testx=np.linspace(-2,2)

        self.fig,self.ax=plt.subplots(figsize=(10,6))
        self.line,=self.ax.plot([],[],label='poly',color='r') #动画plot
        self.text=self.ax.text(0.05,0.95,'',transform=self.ax.transAxes)#动画text

        self.ax.scatter(self.trainx,self.trainy)
        # self.scatter=self.ax.scatter([],[],s=20,marker='*') #动画scatter

    def run(self):
        ani=FuncAnimation(self.fig,self.update,frames=range(2,20),
                          interval=1000,repeat=False)
        plt.show()

    def fit(self,k):
        model=make_pipeline(
            PolynomialFeatures(k),
            LinearRegression(fit_intercept=True))
        model.fit(self.trainx[:,None],self.trainy)

        return model.predict(self.testx[:,None])
    
    def update(self,n):

        yp=self.fit(n)
        self.line.set_data(self.testx,yp) #更新plot
        self.text.set_text(f'degree={n}') #更新text
        # self.scatter.set_offsets([[self.trainx[n],0]])  #更新scatter

        return self.line,self.text
        # return self.line,self.text,self.scatter
    

def main():
    p=Poly()
    p.run()


if __name__ == '__main__':
    main()

