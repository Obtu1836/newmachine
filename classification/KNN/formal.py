import numpy as np
from numpy.linalg import norm
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class Knn:
    def __init__(self,k:int):

        self.k=k

    def fit(self,trainx,trainy):

        self.trainx=trainx
        self.trainy=trainy
    
    def predict(self,test):
        dis=norm(test[:,None]-self.trainx,axis=2) #向量化计算每个测试点到每个训练样本的距离
        # ind=np.argsort(dis,axis=1)[:,:self.k]# 每个测试点都挑出距离自己最近的前k个点
        ind=np.argpartition(dis,self.k,axis=1)[:,:self.k]
        tags=self.trainy[ind]  # 记录前k个点的标签（包含每个测试点的）
        result=stats.mode(tags,axis=1,keepdims=False) # 统计标签的众数
        return result.mode
    
def main():

    x,y=load_iris(return_X_y=True)
    x_train,x_test,y_trian,y_test=train_test_split(x,y,train_size=0.8)

    knn=Knn(15)
    knn.fit(x_train,y_trian)
    res=knn.predict(x_test)
    print(f"{accuracy_score(y_test,res):.3f}")

if __name__ == '__main__':
    main()

    

