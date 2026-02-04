import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class BinaryClusteringAlgorithm:

    def __init__(self, k: int):

        self.k = k

    def _plus(self, x: NDArray, k: int) -> NDArray:

        cents = []
        cents.append(x[np.random.choice(len(x))])

        while len(cents) < k:
            dis2 = np.sum(np.power(x[:, None]-cents, 2), axis=2)
            dis2 = np.min(dis2, axis=1)
            prob = dis2/dis2.sum()
            cum_prob = np.cumsum(prob)
            idx = np.searchsorted(cum_prob, np.random.rand())
            cents.append(x[idx])

        return np.array(cents)

    def kmean(self, x:NDArray, k: int = 2) ->tuple[NDArray,NDArray]:

        centers = self._plus(x, k)
        m, n = x.shape
        tags = np.zeros((m, 2))

        flag = True
        while flag:
            flag = False
            dis = np.sum(np.power(x[:, None]-centers, 2), axis=2)
            mindis = np.min(dis, axis=1)
            min_dis_idx = np.argmin(dis, axis=1)
            if not (tags[:, 0] == min_dis_idx).all():
                flag = True
            tags[:, 0] = min_dis_idx
            tags[:, 1] = mindis

            centers = np.zeros((k, n))
            np.add.at(centers, min_dis_idx, x)

            all_tag_nums = np.bincount(min_dis_idx, minlength=k)
            non_empty_idx = all_tag_nums > 0
            empty_idx = all_tag_nums == 0

            centers[non_empty_idx] /= all_tag_nums[non_empty_idx][:, None]
            if any(empty_idx):
                centers[empty_idx] = x[np.random.choice(m, empty_idx.sum())]

        return centers, tags

    def fit(self, x):
        m, _ = x.shape
        cents = []
        cents.append(x.mean(axis=0))

        record_cents = []
        record_cents.append(cents.copy())

        tags = np.zeros((m, 2))

        record_tags = []
        record_tags.append(tags[:, 0].copy())

        while len(cents) < self.k:
            sse = np.inf
            for i in range(len(cents)):
                current_x = x[tags[:, 0] == i]
                new_center, new_tag = self.kmean(current_x, 2)
                split_sse = new_tag[:, 1].sum()
                non_split_sse = tags[tags[:, 0] != i, 1].sum()
                new_sse = split_sse+non_split_sse

                if new_sse < sse:
                    sse = new_sse
                    sign = i
                    x_ = new_center
                    tag_ = new_tag

            tag_[tag_[:, 0] == 1, 0] = len(cents)
            tag_[tag_[:, 0] == 0, 0] = sign
            tags[tags[:, 0] == sign, :] = tag_

            record_tags.append(tags[:, 0].copy())

            cents.append(x_[1])
            cents[sign] = x_[0]

            record_cents.append(cents.copy())

        return np.array(cents), tags, record_cents, record_tags

    @staticmethod
    def label_match(label_true, label_km):

        d = max(label_true.max(), label_km.max())+1

        w = np.zeros((d, d))
        np.add.at(w, (label_true, label_km), 1)

        ind_true, ind_yp = linear_sum_assignment(w.max()-w)

        mask = np.zeros(d)
        mask[ind_yp] = ind_true

        return mask


class Paint:
    def __init__(self, data, y):

        self.fig, self.ax = plt.subplots(figsize=(10, 6), nrows=1, ncols=2)
        self.fig.suptitle('binary clustering')
        self.data = data
        self.y = y

        self.ax[0].scatter(data[:, 0], data[:, 1], c=y, cmap='viridis',
                           vmin=y.min(), vmax=y.max())

        self.scatter = self.ax[1].scatter(self.data[:, 0], self.data[:, 1],
                                          c=np.zeros(len(data)), vmin=y.min(),
                                          vmax=y.max(), cmap='viridis')

        self.center = self.ax[1].scatter([], [], marker='*', s=200, color='r')

        self.text = self.ax[1].text(
            0.05, 1, '', transform=self.ax[1].transAxes)

    def drawing(self, centers, tags, mask):

        def update(n):

            data = np.array(centers[n])

            tag = tags[n].astype(int)
            change_tag = mask[tag]  # 标签重新映射

            acc = np.sum(change_tag == self.y)/len(self.y)

            self.scatter.set_array(change_tag)  # 对散点动态设置颜色 针对的scatter中c参数
            self.center.set_offsets(data)  # 动态画出散点图
            self.text.set_text(f'{n+1} - iters  acc: {acc:.3f}')  # 动态设置文本

            return self.center, self.scatter, self.text

        self.ax[1].set_xlim(self.data[:, 0].min()-1, self.data[:, 0].max()+1)

        ani = FuncAnimation(self.fig, update, range(len(centers)),
                            interval=2000, blit=False, repeat=False)
        plt.show()


def main():
    k = 5
    x, y = make_blobs(1200, 2, centers=k)[:2]

    model = BinaryClusteringAlgorithm(k)
    _, tags, record_cents, record_tags = model.fit(x)

    yp_label = tags[:, 0].astype(np.int32)

    mask = model.label_match(y, yp_label)

    paint = Paint(x, y)
    paint.drawing(record_cents, record_tags, mask)


if __name__ == '__main__':
    main()
