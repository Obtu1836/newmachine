import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from classification.DecisionTree.formal import DecisionTree,Node

np.set_printoptions(2,suppress=True)
def make_data(xmax,ymax):
    step=4
    x=np.arange(0,stop=xmax,step=step,dtype=float)
    y=np.arange(0,stop=ymax,step=step,dtype=float)
    xx,yy=np.meshgrid(x,y)
    xx[1:-1,1:-1]+=np.random.uniform(0,1,(int(np.ceil(xmax/step))-2,int(np.ceil(ymax/step)-2)))
    yy[1:-1,1:-1]+=np.random.uniform(0,1,(int(np.ceil(ymax/step))-2,int(np.ceil(xmax/step)-2)))
    data=np.stack([xx,yy],axis=2).reshape(-1,2)
    condition1=((0<=data[:,0])&(data[:,0]<42))&((100<=data[:,1])&(data[:,1]<ymax))
    condition2=(((100<=data[:,0])&(data[:,0]<xmax))&((40<data[:,1])&(data[:,1]<100)))
    idx=np.where(condition1|condition2)[0]
    tags=np.zeros(len(data),dtype=int)
    tags[idx]=1
    return data,tags


class New(DecisionTree):

    def __init__(self,max_depth,mode):
        super().__init__(mode,max_depth)

    def get_path(self,xmin,xmax,ymin,ymax):
        paths=[]
        levels=[]
        def inner(node:Node,xmin,xmax,ymin,ymax,level):

            if node.leaf is not None:
                return 
            
            if node.column==1:
                conts=(node.column,node.value,xmin,xmax)
                paths.append(conts)
                levels.append(level)
                if node.left is not None:
                    inner(node.left,xmin,xmax,ymin,node.value,level+'L')
                if node.right is not None:
                    inner(node.right,xmin,xmax,node.value,ymax,level+'R')
            else:
                conts=(node.column,node.value,ymin,ymax)
                paths.append(conts)
                levels.append(level)
                if node.left is not None:
                    inner(node.left,xmin,node.value,ymin,ymax,level+'L')
                if node.right is not None:
                    inner(node.right,node.value,xmax,ymin,ymax,level+'R')

        inner(self.tree,xmin,xmax,ymin,ymax,'O') #type: ignore
    
        return paths,levels
    
class Vision:
    def __init__(self):
        
        self.fig,self.ax=plt.subplots(figsize=(10,6))

        self.model=New(5,'entropy')
    
    def draw_scatter(self,data,tag):

        scatter=data[tag==0]
        marker=data[tag==1]

        self.ax.scatter(scatter[:,0],scatter[:,1])
        self.ax.scatter(marker[:,0],marker[:,1],marker='*',s=80)

    def draw_line(self,data,tag):

        self.ax.set_xlim(-3,143)
        self.ax.set_ylim(-3,143)
        self.draw_scatter(data,tag)
        self.model.fit(data,tag)
        paths,levels=self.model.get_path(0,140,0,140)

        def update(n):

            cont=paths[n] # column value xmin/ymin  xamx/ymax
            level=levels[n]
            if cont[0]==0:  #column value ymin ymax
                self.ax.vlines(*cont[1:],color='r',lw=5) #竖直线
                if level[-1]=='O' or level[-1]=='R':
                    self.ax.text(cont[1],cont[3],level)
                else:
                    self.ax.text(cont[1],cont[2],level)
                
            else: # column value xmin,xmax
                self.ax.hlines(*cont[1:],color='r',lw=5)
                if level[-1]=='O' or level[-1]=='R':
                    self.ax.text(cont[3],cont[1],level)
                else:
                    self.ax.text(cont[2],cont[1],level)
            return [self.ax]
        
        ani=FuncAnimation(self.fig,update,range(len(paths)),
                          repeat=False,interval=1000)
        plt.show()

    def draw(self,data,tag):

        self.draw_line(data,tag)


def main():
    data,tags=make_data(140,140)

    vis=Vision()
    vis.draw(data,tags)

    
if __name__ == '__main__':
    main()

