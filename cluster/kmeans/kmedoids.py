import numpy as np 
from numpy.typing import NDArray
from numpy.linalg import norm
from cluster.km.roulette import roulette_method
from scipy.spatial.distance import pdist,squareform
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

def kmedoid(data:NDArray, k: int) -> tuple[NDArray, NDArray]:
    """
    K-Medoids 聚类算法实现
    :param data: 输入的数据集 (m, n)
    :param k: 聚类簇数
    :return: (最终质心, 标签与距离信息)
    """
    # 1. 初始化：使用轮盘赌法选择初始 k 个中心点（Medoids）
    cents = roulette_method(data, k)
    m, n = data.shape
    # tags 用于存储：第一列为所属簇索引，第二列为到所在簇中心的距离
    tags = np.zeros((m, 2))

    flag = True
    while flag:
        flag = False

        # 2. 分配阶段：计算每个样本到所有中心点的欧式距离
        # data[:, None] 形状为 (m, 1, n)，cents 形状为 (k, n)
        # 利用广播机制得到 (m, k) 的距离矩阵
        dis = norm(data[:, None] - cents, axis=2)
        
        # 找到最近中心的索引及距离值
        disarg = np.argmin(dis, axis=1) # (m,) 每个点属于哪个簇
        disval = np.min(dis, axis=1)    # (m,) 最小距离

        # 如果当前分配结果与上一轮不同，则继续迭代
        if not (tags[:, 0] == disarg).all():
            flag = True
        
        tags[:, 0] = disarg
        tags[:, 1] = disval

        # 3. 更新阶段：在每个簇内寻找新的代表点（Medoid）
        for i in range(k):
            # 提取属于第 i 个簇的所有样本点
            dat = data[tags[:, 0] == i]
            if np.size(dat) != 0:
                # 调用 cal_idx 找到簇内到其他点距离之和最小的点作为新中心
                idx = cal_idx(dat)
                cents[i] = dat[idx]
            else:
                # 如果某个簇空了，则随机从原始数据中选一个点作为新中心
                cents[i] = data[np.random.choice(m)]
    return cents, tags

def cal_idx(data: NDArray) -> int:
    """
    计算 Medoid：在当前集合中找到一个点，使其到集合内所有其他点的距离之和最小
    """
    # 计算集合内两两之间的欧式距离（返回压缩后的向量形式）
    dist_vec = pdist(data, metric='euclidean') 
    # 将压缩距离向量转换为对称方阵 (len(data), len(data))
    dist_matrix = squareform(dist_vec)
    # 按行求和，得到每个点到其他所有点的总距离
    sum_dis = dist_matrix.sum(axis=1)
    # 返回总距离最小的点的索引
    return int(np.argmin(sum_dis))# 显示的转化


def main():
    # 测试参数：4个簇
    k = 4
    # 生成 200 个样本点，2个特征，包含 k 个中心点的数据集
    data, labels = make_blobs(200, 2, centers=k)[:2]

    # 执行 K-Medoids 聚类
    cents, tags = kmedoid(data, k)

    # 可视化结果
    # 绘制原始数据点，颜色根据 make_blobs 生成的真实标签区分
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    # 绘制聚类计算出的质心（红色星号，大小为120）
    plt.scatter(cents[:, 0], cents[:, 1], marker='*', s=120, color='r')
    plt.show()

    # print(cents)

if __name__ == '__main__':
    main()




