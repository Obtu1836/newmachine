import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve,auc
from metric.classification.mul_class_pr import make_data

def cal_class(probas,bin_targets):
    num_class=probas.shape[1]

    for i in range(num_class):
        fpr,tpr,_=roc_curve(bin_targets[:,i],probas[:,i])
        area=auc(fpr,tpr)
        plt.plot(fpr,tpr,label=f'class{i}{round(area,3)}')
    label=roc_auc_score(bin_targets,probas,average='macro')
    plt.title(f'macro_auc:{round(label,3)}')
    plt.legend(loc='best')
    plt.show()

def main():
    num_class=4
    probas,bin_target=make_data(num_class,alpha=0.5)
    cal_class(probas,bin_target)

if __name__ == '__main__':
    main()
