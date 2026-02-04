import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
from sklearn.datasets import load_digits,fetch_openml
from sklearn.model_selection import cross_validate,StratifiedKFold
from sklearn.base import ClassifierMixin,BaseEstimator
from sklearn.naive_bayes import MultinomialNB   

class NaiveBayes(ClassifierMixin,BaseEstimator):

    def __init__(self, alpha=1):

        self.alpha = alpha #用于平滑的数
        '''ordianEncoder LabelEncoder 都是用于将离散型或者类别型数据
        转换为 连续的整数 (从0开始) 
        ordianencoder 用于二维数组 labelencoder用于一维数组

        它将每个特征的唯一取值映射为整数 如果某个特征有取值 ['A', 'B', 'C']
        它会将其转换为 [0, 1, 2]  这样还有一个好处是 便于索引

        '''
        self.coder = OrdinalEncoder(handle_unknown='use_encoded_value',
                                    unknown_value=-1, dtype=int)
        self.ycoder=LabelEncoder()

    def _stat_perfeature_univalue(self, df: pd.DataFrame):
        '''统计每个类别的数组 在每个数组中 每列中每个特征值的个数'''
        nums = df.apply(lambda x: x.value_counts(), axis=0)
        return nums.T

    def fit(self, x:NDArray, y:NDArray):

        trainx = self.coder.fit_transform(x)
        y=np.asarray(self.ycoder.fit_transform(y))

        _, n_features = trainx.shape # 确认特征数
        column_name = ['x'+str(x) for x in range(1, n_features+1)] #设列名 方便看
        df = pd.DataFrame(trainx, columns=column_name) #建立dataframe


        '''
        groups 存储了每个类别中，每个特征下，各特征值出现的频数。
        它是一个 DataFrame,具有两层索引 (类别, 特征)。
        列名 (columns) 是特征经过编码后的整数值 (0, 1, 2...)。
        元素值表示：在特定类别下，某个特征取特定值的次数。
        '''
        groups = df.groupby(y).apply(self._stat_perfeature_univalue)
        groups.fillna(0, inplace=True)

        self.classes_, nums = np.unique(y, return_counts=True)# 统计类别和个数
        class_num = len(self.classes_) # 计算类别数量

        '''
        统计每个特征分别有几个特征值 这关系到计算条件概率的分母
        计算出以后再进行扩展 方便 每个类别都能用到 需要保证索引能够对齐 
        
        '''
        num_features = df.apply(lambda x: len(np.unique(x)), axis=0)#统计所有训练数据 每个特征有几个特征值
        num_features = pd.Series(
            np.tile(num_features, class_num), index=groups.index) #扩展 有几类 扩展几类

        '''计算条件概率：
        分子：频数 + 平滑项 alpha
        分母：该类别总样本数 + alpha * 该特征的取值种类数
        '''
        model = (groups+self.alpha).div(groups.sum(axis=1) +
                                        self.alpha*num_features, axis=0)
        

        logmodel = np.log(model.values) # 转log 原因是为了后续使用加法运算 不转log就得用乘法
        self.log_model = logmodel.reshape(class_num, n_features, -1)# 转numpy 方便后续 使用高级索引 向量化方式取值

        y_prob = (nums+self.alpha)/(nums.sum()+class_num*self.alpha)# 计算先验概率
        self.log_y_prob = np.log(y_prob)# 转log 理由同上

        return self

    def predict(self, test:NDArray):

        test = np.atleast_2d(test) # 如果是1维(d,) 自动转为(1,d) 2维以上保持不变
        test = self.coder.transform(test)
        n_features = test.shape[1]
        '''通过合理的构造索引 向量化方式取值 速度快'''
        joint_proba = self.log_model[:, np.arange(n_features), test].sum(axis=2)
        # joint_proba 的形状是 (类别数, 样本数)
        joint_proba = joint_proba + self.log_y_prob[:, None] 
        # argmax(axis=0) 取出的是每个样本对应的最大类别索引
        idx = np.argmax(joint_proba, axis=0)

        return self.ycoder.inverse_transform(self.classes_[idx])


def main():

    x,y=load_digits(return_X_y=True)
    x=np.asarray(x)
    model=NaiveBayes(1)
    # model=MultinomialNB(alpha=1)

    cv_style=StratifiedKFold(5,shuffle=True)
    

    res=cross_validate(model,x,y,cv=cv_style,scoring='accuracy')
    print(f"acc: {res['test_score'].mean():.3f}")
    


if __name__ == '__main__':
    main()
