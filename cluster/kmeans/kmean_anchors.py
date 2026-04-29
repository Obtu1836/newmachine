import numpy as np 
from numpy.typing import NDArray

def update_cents(boxes: NDArray, cents: NDArray, k: int) -> tuple[NDArray, float]:

    N, M = boxes.shape
    bw = boxes[:, 0][:, None]  # (N,1)
    bh = boxes[:, 1][:, None]

    cw = cents[:, 0][None, :]  # (1,k)
    ch = cents[:, 1][None, :]

    inw = np.minimum(bw, cw)  # (N,K)
    inh = np.minimum(bh, ch)  # (N,K)
    in_area = inw*inh  # (N,K)

    barea = bw*bh  # (N,1)
    carea = cw*ch  # (1,K)
    union = barea+carea-in_area  # (N,K)
    iou = in_area/(union+1e-8)
    dist = 1-iou  # (N,K)

    min_dist_ind = np.argmin(dist, axis=1)  # (N,)
    mid_dist = dist[np.arange(len(min_dist_ind)), min_dist_ind]
    loss = mid_dist.sum()

    counts = np.bincount(min_dist_ind, minlength=k)
    new_cents = np.zeros((k, M))
    np.add.at(new_cents, min_dist_ind, boxes)

    empty = counts == 0
    not_empty = counts > 0
    new_cents[not_empty] /= counts[not_empty][:, None]

    if np.any(empty):
        num = empty.sum()
        new_cents[empty] = boxes[np.random.choice(N, num)]

    return new_cents, loss


def anchor_box_kmean(boxes: NDArray, k: int, rtol: float,
                     max_iters: int = 20000):

    N = len(boxes)

    cents = boxes[np.random.choice(N, k)]

    init_loss = np.inf
    i = 1
    while True:
        cents, loss = update_cents(boxes, cents, k)
        i += 1
        if abs(loss-init_loss) < rtol or i > max_iters:
            break
        init_loss = loss

    return cents

def main():
    k=5
    boxes=np.random.randint(0,2000,size=(450,2))
    anchors=anchor_box_kmean(boxes,k,1e-6)
    print(anchors)

if __name__ == '__main__':
    main()
