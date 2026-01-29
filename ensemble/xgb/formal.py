import xgboost as xgb
import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class X:
    def __init__(self, num_calss: int, **params):

        default_params = {'booster': 'gbtree',
                          'nthread': 4,
                          'device': 'cpu',
                          'eta': 0.1,
                          'max_depth': 5,
                          'objective': 'multi:softprob',
                          'eval_metric': 'mlogloss',
                          'num_class': num_calss}

        self.params = default_params | params
        self.num_calss = num_calss

    def _get_best_iter(self, data):

        it = xgb.cv(self.params, 
                    data, 
                    metrics='mlogloss',
                    stratified=True, 
                    nfold=10, 
                    num_boost_round=1000)
        
        df = pd.DataFrame(it)

        best_iter = df['test-mlogloss-mean'].argmin()
        return best_iter+1

    def fit(self, dtrain, dval):
        best_iter = self._get_best_iter(dtrain)
        evals = [(dtrain, 'train'), (dval, 'eval')]
        self.model = xgb.train(self.params, 
                               dtrain, 
                               num_boost_round=best_iter,
                               evals=evals, 
                               early_stopping_rounds=5, 
                               verbose_eval=20)
        '''
self.model 常用方法：
predict(data): 对传入的 DMatrix 数据进行预测。
get_score(importance_type='weight'): 获取特征重要性。可选类型包括:
'weight': 特征出现在树中的次数。
'gain': 该特征带来的平均增益。
'cover': 该特征覆盖的平均样本数。
get_fscore(): 获取特征重要性原始字典（等同于 get_score(importance_type='weight')）。
save_model(fname): 将模型保存到文件。
load_model(fname): 从文件加载模型。
dump_model(fout): 将模型导出为文本或 JSON 格式。
        '''


        return self

    def predict_proba(self, test):
        
        res=self.model.predict(test)

        return res

    def predict(self,test):

        proba=self.predict_proba(test)
        label=np.argmax(proba,axis=1)
        return label
    
    def feature_importance(self):

        importantce=self.model.get_fscore()
        col=sorted(importantce.items(),key=lambda x:x[1])[-5:][::-1]
        
        return [x[0] for x in col]

def main():
    x, y = load_wine(return_X_y=True)
    x = np.asarray(x)

    trainx, test_valx, trainy, test_valy = train_test_split(
        x, y, train_size=0.6, shuffle=True)
    valx, testx, valy, testy = train_test_split(
        test_valx, test_valy, train_size=0.5, shuffle=True)

    num_calss = len(np.unique(y))
    model = X(num_calss)

    model.fit(xgb.DMatrix(trainx, label=trainy),
              xgb.DMatrix(valx, label=valy))
    
    yp=model.predict(xgb.DMatrix(testx))
    print(f"acc:{accuracy_score(testy,yp)}")
    print(model.feature_importance())

    

if __name__ == '__main__':
    main()
