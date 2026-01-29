import numpy as np
from numpy.typing import NDArray
from typing import Literal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from classification.DecisionTree.non_prune import DecisionTree,Node


np.set_printoptions(precision=2,suppress=True)
np.random.seed(42)

class New(DecisionTree):
    def __init__(self,max_depth:int,mode:Literal['entropy','gini']):
        super().__init__(max_depth,mode)
    
    def make_data(self,x_min:int,x_max:int,y_min:int,y_max:int):

        x=np.arange(x_min,x_max,3,dtype=float)
        y=np.arange(y_min,y_max,3,dtype=float)
        xs,ys=np.meshgrid(x,y)
        data=np.stack([xs,ys],axis=2).reshape(-1,2)
        data+=np.random.randn(len(data),2)*0.2
        x_min,x_max=data[:,0].min(),data[:,0].max()
        y_min,y_max=data[:,1].min(),data[:,1].max()
        condition1=((x_min<data[:,0])&(data[:,0]<42))&((100<data[:,1])&(data[:,1]<y_max))
        condition2=(((100<data[:,0])&(data[:,0]<x_max))&((40<data[:,1])&(data[:,1]<100)))
        idx=np.where(condition1|condition2)[0]
        tags=np.zeros(len(data),dtype=int)
        tags[idx]=1
        return data.round(2),tags
    
    def paint(self,data,tags,p,texts):

        fig,ax=plt.subplots(figsize=(10,6))
        circle=data[tags==0]
        star=data[tags==1]
        ax.scatter(circle[:,0],circle[:,1])
        ax.scatter(star[:,0],star[:,1],marker='*',s=40)

        def update(n):
            
            content=p[n]
            text=texts[n]
            a,b,c,d=content
            x,y,t=text
            if a==0:
                ax.vlines(b,c,d,color='g',linewidths=5)
            else:
                ax.hlines(b,c,d,color='g',lw=5)

            ax.text(x,y,t,color='r',fontsize=20)
            return [ax]

        ani=FuncAnimation(fig,update,range(len(p)),
                          repeat=False,interval=1000)

        plt.show()

    def get_path(self,x_min,x_max,y_min,y_max):

        path=[]
        texts=[]
        def fun(node:Node,x_min,x_max,y_min,y_max,level):

            if node.leaf is not None:
                return 

            if node.column==0:
                content=(node.column,node.value,y_min,y_max)
                path.append(content)
                sinal=level[-1]
                if sinal=='R' or sinal=='O':
                    texts.append((node.value,y_max,level))
                else:
                    texts.append((y_min,node.value,level))
                if node.left is not None:
                    fun(node.left,x_min,node.value,y_min,y_max,level+'-L')
                if node.right is not None:
                    fun(node.right,node.value,x_max,y_min,y_max,level+'-R')
            else:
                content=(node.column,node.value,x_min,x_max)
                path.append(content)
                if level[-1]=='R' or level[-1]=='O':
                    texts.append((x_max,node.value,level))
                else:
                    texts.append((x_min,node.value,level))
                if node.left is not None:
                    fun(node.left,x_min,x_max,y_min,node.value,level+'-L')
                if node.right is not None:
                    fun(node.right,x_min,x_max,node.value,y_max,level+'-R')
            
        fun(self.tree,x_min,x_max,y_min,y_max,'O')
        
        return path,texts

def main():

    new=New(4,'entropy')
    x_min,x_max,y_min,y_max=0,140,0,140
    data,tags=new.make_data(x_min,x_max,y_min,y_max)
    x_min,x_max=data[:,0].min(),data[:,0].max()
    y_min,y_max=data[:,1].min(),data[:,1].max()
    new.fit(data,tags)

    paths,texts=new.get_path(x_min,x_max,y_min,y_max)
    
    new.paint(data,tags,paths,texts)
    new.print_path('O')


if __name__ == '__main__':
    main()





