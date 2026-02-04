import numpy as np
import joblib
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve,f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin,BaseEstimator,clone

'''
目的：
我想对一个二分类任务 实现 通过调整阈值 找到一个最好的f1_score,因为 f1_score
结合了 precision和recall 通过f1_score 这个确定数值来反映模型的能力

'''

class Model(ClassifierMixin,BaseEstimator):
    '''
    Model 的 Docstring
    自定义一个模型 这个模型的作用就是 包裹 一个分类器和一个参数 这个参数在本例中就是阈值
    如果不采取这种方式 那么当从训练集中 找到合适的阈值以后 只能在原分类器中手动修改代码

    采取这种方式 可以在执行过程中 自动把参数注入进去并修改
    '''

    def __init__(self,estimator,threshold=0.5):

        self.estimator=estimator
        self.threshold=threshold
    
    def fit(self,x,y):
        self.classes_=np.unique(y)
        '''
clone 来自 sklearn.base,用于创建一个新的、未训练的估计器实例：保留超参数（通过 get_params),但不复制训练后产生的属性（如 coef_, tree_, fitted_ 等）。
对包含子估计器的结构会递归克隆(deep clone 参数语义）,因此不会共享已训练的内部状态。
,clone(self.estimator) 的作用是生成一个干净的副本来 fit,避免修改或重用原始 estimator 的已训练状态。
        '''
        self.estimator_=clone(self.estimator)# 克隆模型
        self.estimator_.fit(x,y)
        return self
    
    def predict_proba(self,X):
        proba=self.estimator_.predict_proba(X)
        return np.asarray(proba[:,1])
    
    def predict(self,X):

        proba=self.predict_proba(X)
        label=np.where(proba>self.threshold,1,0)
        return label


def fbeta_score(beta,precision,recall):
    "计算fbeta  score"
    molecular=(1+beta**2)*recall*precision
    denominator=beta**2*precision+recall

    return np.divide(molecular,denominator+1e-6)


def train(model,xtrain,ytrain,xval,yval):
    '''
    初始在训练集上 训练原始数据 并将训练出来的模型 通过在验证集上 进行评估 
    然后通过precision_recall_curve函数 找到 precision recall threshold
    根据前2个返回值 计算fscore 并找到最大值 返回fscore最大时 阈值的值
    '''

    model.fit(xtrain,ytrain)
    y_prob=model.predict_proba(xval)

    precision,recall,threshold=precision_recall_curve(yval,y_prob)
    if len(threshold)==0:
        return model.threshold

    f_betascore=fbeta_score(1,precision[:-1],recall[:-1])
    idx=np.argmax(f_betascore)
    best_threshold=threshold[idx]
    return best_threshold


def evaluation(model, xtest, ytest):
    '''在测试集 评估模型在当前阈值下的真实表现
    这里不再使用 cross_validate 因为cross_validate 会调用fit 
    又会重新训练一次
    '''
    y_pred = model.predict(xtest)
    f1 = f1_score(ytest, y_pred)
    acc = accuracy_score(ytest, y_pred)
    print(f"F1 Score: {f1:.3f}")
    print(f"Accuracy: {acc:.3f}")


def save_model(model):
    base_path=Path(__file__).resolve().parents[1]
    # print(base_path)
    save_path=base_path /'models'/ 'best_threshold.pkl'
    joblib.dump(model,save_path)
    print('保存结束！')


def main():

    global seed
    seed=20
    x,y=load_breast_cancer(return_X_y=True)
    x=np.asarray(x)
    '''
    为了避免数据泄露 分为训练集 验证集 测试集   训练集用于训练模型 验证集用于找到最佳的阈值 最后在测试集
    运行 查看模型表现
    '''
    xtrain,xtestval,ytrain,ytestval=train_test_split(x,y,train_size=0.6,random_state=seed)
    xval,xtest,yval,ytest=train_test_split(xtestval,ytestval,train_size=0.5,random_state=seed)
    scaler=MinMaxScaler()
    base_estinmator=GradientBoostingClassifier(loss='log_loss',random_state=seed)
    clf=Model(base_estinmator,0.5)

    model=Pipeline([('scaler',scaler),('clf',clf)])#存放的引用 修改了模型 model内部的对象也会改变

    
    # 找到最佳阈值前，先训练一次以确保 model 内部已 fit（针对旧逻辑的评估）
    model.fit(xtrain, ytrain) 
    print('默认阈值下 指标分数：')
    evaluation(model, xtest, ytest)

    # 寻找并设置新阈值
    best_threshold = train(model, xtrain, ytrain, xval, yval)
    model.set_params(clf__threshold=best_threshold)
    
    print(f"阈值更换为 {best_threshold:.4f}")
    print('更换阈值后 指标的评价分数：')
    # 此时 model 内部的 estimator_ 已经是基于 xtrain 训练好的，直接预测即可
    evaluation(model, xtest, ytest) #

    save_model(model)

   


if __name__ == '__main__':
    main()







