import numpy as np 

from sklearn.metrics import confusion_matrix
'''
1v1 不常用于最终评分 使用场景 有两类
    1 深度分析 某2个类之间的关系 比如 预测和实际的中 两个类总会出现混乱时
    2 某些特定算法实现 比如svm 在多分类时 内部可能采用1v1策略
'''

label=np.array([0,1,2,0,1,1,2,1],dtype=int)

yp=np.array([0,1,1,2,1,0,2,1],dtype= int)


# 假设已经得到了完整的 3x3 混淆矩阵
conf = confusion_matrix(label, yp)

def get_1vs1_matrix(conf, class_i, class_j):
    # 提取交叉点形成 2x2 矩阵
    # [[TP_ii,  FP_ij],
    #  [FN_ji, iTP_jj ]]
    matrix_1v1 = np.array([
        [conf[class_i, class_i], conf[class_i, class_j]],
        [conf[class_j, class_i], conf[class_j, class_j]]
    ])
    return matrix_1v1

# 示例：查看类别 0 和 类别 1 之间的互打情况
m_01 = get_1vs1_matrix(conf, 0, 1)
print(f"0 vs 1 的 1v1 矩阵:\n{m_01}")

# 计算在此子集下的 recall (针对类别 0)
recall_0 = m_01[0, 0] / m_01[0, :].sum() if m_01[0, :].sum() > 0 else 0