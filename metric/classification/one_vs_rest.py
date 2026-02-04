import numpy as np 
from sklearn.metrics import (multilabel_confusion_matrix,
                             confusion_matrix,
                             recall_score,
                             precision_score)


def sk_method(yt,yp):
    '''
    sk_method 的 Docstring
    
    :param yt: 说明
    :param yp: 说明
    利用sklearn 根据标签 直接生成 k个分类的 二分类混淆矩阵
    '''
    mcm=multilabel_confusion_matrix(yt,yp)
    return mcm

def self(confusion):
    '''
    self 的 Docstring
    
    :param confusion: 说明
    从现有的多分类混淆矩阵 提取 各个类别的的二分类混淆矩阵 
    通过这些二分类矩阵 可以进一步算出  recall,precision等指标

    混淆矩阵 遵循 axis=0 方向 预测标签  axis=1 方向 样本标签
    '''
    num_calss=confusion.shape[0]
    results=np.stack([np.zeros((2,2),dtype=int)]*3,axis=0)


    recalls=[]
    precisions=[]
    for i in range(num_calss):

        tp=confusion[i,i]
        fn=confusion[i,:].sum()-tp
        fp=confusion[:,i].sum()-tp
        tn=confusion.sum()-tp-fn-fp

        mini_confusion=np.array([[tn,fp],
                                 [fn,tp]],dtype=np.int32)
        recall=tp/(tp+fn) # 计算召回率
        recalls.append(recall)

        precision=tp/(tp+fp)   #计算精确率
        precisions.append(precision)
        results[i,...]=mini_confusion
    
    return results,np.array(recalls),np.array(precisions)


def main():


    ytrue=np.array([0,1,1,2,0,1,0,1,0,2])
    ypred=np.array([1,0,2,0,1,1,0,1,0,2])

    confusion=confusion_matrix(ytrue,ypred)
    # print(f"confusion:\n",confusion)

    p1=sk_method(ytrue,ypred)
    p2,macro_recall,macro_precision=self(confusion)

    print((p1==p2).all())

    rel=recall_score(ytrue,ypred,average='macro') # 手动实现macro方法
    pre=precision_score(ytrue,ypred,average='macro')


   
    print(rel==macro_recall.mean())  # 比较recall 是否一样  macro 下
    print(pre==macro_precision.mean()) # 比较precision是否一样 macro下

    micro_recall=recall_score(ytrue,ypred,average='micro')

    micro=p2.sum(axis=0)
    micro_recall_self=micro[1,1]/micro[1].sum()  #手动实现 micro方法

    print(micro_recall_self==micro_recall) #比较 recall是否一样 micro下


if __name__ == '__main__':
    main()




