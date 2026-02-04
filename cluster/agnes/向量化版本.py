import numpy as np 
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

def opts(x, k):
    n = x.shape[0]
    # 1. 预计算所有点对之间的距离矩阵 (N x N)
    # 这是单链接聚类的基础，后续只需在矩阵上操作
    dist_matrix = squareform(pdist(x))
    # 将对角线设为无穷大，避免自己与自己合并
    np.fill_diagonal(dist_matrix, np.inf)

    # 2. 初始化：每个点一个簇，记录每个簇包含的原始索引
    clusters = [[i] for i in range(n)]
    # 记录哪些行/列是有效的（模拟 pop 操作，但更高效）
    active_nodes = list(range(n))

    while len(active_nodes) > k:
        # 3. 在当前的距离矩阵中寻找最小值及其索引
        # 只在活跃的行和列中寻找
        current_dist = dist_matrix[np.ix_(active_nodes, active_nodes)]
        min_idx = np.argmin(current_dist)
        i_idx, j_idx = np.unravel_index(min_idx, current_dist.shape)
        
        # 转换为原始索引
        u = active_nodes[i_idx]
        v = active_nodes[j_idx]

        # 4. 更新距离矩阵 (单链接准则)
        # 合并后新簇到其他簇的距离 = min(dist[u, h], dist[v, h])
        # 把 u 当作合并后的新簇，更新 u 行和 u 列
        dist_matrix[u, :] = np.minimum(dist_matrix[u, :], dist_matrix[v, :])
        dist_matrix[:, u] = dist_matrix[u, :]
        dist_matrix[u, u] = np.inf # 保持对角线

        # 5. 合并簇信息并移除已合并的节点 v
        clusters[u].extend(clusters[v])
        active_nodes.pop(j_idx) # 从活跃列表中移除 v

    # 6. 根据索引还原数据点
    return [x[clusters[idx]] for idx in active_nodes]

def main():
    x, y = make_moons(500, noise=0.05)
    
    # 传入原始 numpy 数组
    res = opts(x, 2)

    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(x[:, 0], x[:, 1], c=y)
    ax[0].set_title("Original")
    
    a, b = res
    ax[1].scatter(a[:, 0], a[:, 1], color='r')
    ax[1].scatter(b[:, 0], b[:, 1], color='g')
    ax[1].set_title("Optimized Manual Implementation")
    plt.show()

if __name__ == '__main__':
    main()




