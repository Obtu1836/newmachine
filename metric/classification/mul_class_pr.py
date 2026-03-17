import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_recall_curve,average_precision_score,auc

'''
precision_recall_curve: 输入二值化的标签 和预测概率 返回precision,recall,h和阈值
auc 计算曲线下面积 通用函数
avgrage_precision 也是计算曲线面积 是计算总的 通过参数avg.=macro,micro等实现
'''

def make_data(num_class:int,alpha=0.5):

    binarizer=LabelBinarizer()

    target=np.random.randint(0,num_class,2400)
    bin_target=np.asarray(binarizer.fit_transform(target))
    probas=bin_target+np.random.randn(*bin_target.shape)*alpha
    probas=softmax(probas,axis=1)

    return probas, bin_target

def cal_class(probas,bin_target):
    k=probas.shape[1]
    for i in range(k):
        pre,rec,_=precision_recall_curve(bin_target[:,i],
                                         probas[:,i])
        area=auc(rec,pre)# x轴 y轴
        plt.plot(rec,pre,label=f'class{i}{round(area,3)}')
    ap=average_precision_score(bin_target,probas,average='macro')
    plt.title(f"macro_ap:{round(ap,3)}")
    plt.legend(loc='best')
    plt.show()

def main():
    num_calss=4
    probas,bin_target=make_data(num_calss)

    cal_class(probas,bin_target)
    

if __name__ == '__main__':
    main()

