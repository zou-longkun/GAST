import torch
import torch.nn.functional as F
import numpy as np


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  # B x npoint
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    centroids_vals = torch.zeros(B, npoint, C).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest  # save current chosen point index
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  # get the current chosen point value
        centroids_vals[:, i, :] = centroid[:, 0, :].clone()
        dist = torch.sum((xyz - centroid) ** 2, 2)  # euclidean distance of points from the current centroid
        mask = dist < distance  # save index of all point that are closer than the current max distance
        distance[mask] = dist[mask]  # save the minimal distance of each point from all points that were chosen until now
        farthest = torch.max(distance, -1)[1]  # get the index of the point farthest away
    return centroids, centroids_vals


def cal_loss(pred, ground_truth, smoothing=True):
    """
    Calculate cross entropy loss, apply label smoothing if needed.
    """

    ground_truth = ground_truth.contiguous().view(-1).long()

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, ground_truth.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, ground_truth, reduction='mean')

    return loss


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball query.
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]

    Output:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(k, xyz, new_xyz):
    """
    K nearest neighborhood.
    Input:
        k: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]

    Output:
        group_idx: grouped points index, [B, S, k]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, k, dim=-1, largest=False, sorted=False)
    return group_idx


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]

    Output:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def sample_and_ball_group(s, radius, n, coords, features):
    """
    Sampling by FPS and grouping by ball query.
    Input:
        s[int]: number of points to be sampled by FPS
        k[int]: number of points to be grouped into a neighbor by ball query
        n[int]: fix number of points in ball neighbor
        coords[tensor]: input points coordinates data with size of [B, N, 3]
        features[tensor]: input points features data with size of [B, N, D]

    Returns:
        new_coords[tensor]: sampled and grouped points coordinates by FPS with size of [B, s, k, 3]
        new_features[tensor]: sampled and grouped points features by FPS with size of [B, s, k, 2D]
    """
    batch_size = coords.shape[0]
    coords = coords.contiguous()

    # FPS sampling
    fps_idx, _ = farthest_point_sample(coords, s)  # [B, s]
    new_coords = index_points(coords, fps_idx)  # [B, s, 3]
    new_features = index_points(features, fps_idx)  # [B, s, D]

    # ball_query grouping
    idx = query_ball_point(radius, n, coords, new_coords)  # [B, s, n]
    grouped_features = index_points(features, idx)  # [B, s, n, D]

    # Matrix sub
    grouped_features_norm = grouped_features - new_features.view(batch_size, s, 1, -1)  # [B, s, n, D]

    # Concat, my be different in many networks
    aggregated_features = torch.cat([grouped_features_norm, new_features.view(batch_size, s, 1, -1).repeat(1, 1, n, 1)],
                                    dim=-1)  # [B, s, n, 2D]

    return new_coords, aggregated_features  # [B, s, 3], [B, s, n, 2D]


def sample_and_knn_group(k, features, lg, hard=False):
    """
    Sampling by gumbel_softmax and grouping by KNN.
    Input:
        k[int]: number of points to be grouped into a neighbor by KNN
        coords[tensor]: input points coordinates data with size of [B, N, 3]
        features[tensor]: input points features data with size of [B, N, D]
        lg[tensor]: logits data with size of [B, N, S]
        hard[bool]: gumbel sampling with soft or hard

    Returns:
        new_features[tensor]: sampled and grouped points features by FPS with size of [B, s, k, 2D]
    """
    batch_size, s = lg.shape[0], lg.shape[1]

    # gumbel sampling
    # y = gumbel_softmax(lg, 0.2, hard)  # y:[B, s, N]
    # new_features = torch.bmm(y, features)

    fps_idx, _ = farthest_point_sample(features, s)  # [B, s]
    new_features = index_points(features, fps_idx)  # [B, s, D]

    # K-nn grouping
    idx = knn_point(k, features, new_features)  # [B, s, k]
    grouped_features = index_points(features, idx)  # [B, s, k, D]

    # Matrix subtraction
    grouped_features_norm = grouped_features - new_features.view(batch_size, s, 1, -1)  # [B, s, k, D]

    # Concat
    aggregated_features = torch.cat([grouped_features_norm, new_features.view(batch_size, s, 1, -1).repeat(1, 1, k, 1)], dim=-1)  # [B, s, k, 2D]

    return aggregated_features  # [B, s, k, 2D]


# Down sampling: critical point layer
def CPL(x, s):
    """
    Input:
        x: points feature [N, C]
        s: down sampling num
    Return:
        f_out: down sampled points feature, [M, C]
    """
    fs = np.array([])
    fr = np.array([]).astype(int)
    fmax = x.max(0)
    idx = x.argmax(0)
    _, d = np.unique(idx, return_index=True)
    uidx = np.argsort(d)
    for i in uidx:
        mask = (i == idx)
        val = fmax[mask].sum()
        fs = np.append(fs, val)
        fr = np.append(fr, mask.sum())
    sidx = np.argsort(-fs)
    suidx = uidx[sidx]
    fr = fr[sidx]
    midx = np.array([]).astype(int)
    t = 0
    for i in fr:
        for j in range(int(i)):
            midx = np.append(midx, suidx[t])
        t += 1
    rmidx = np.resize(midx, s)
    fout = x[rmidx]
    print(midx, rmidx)
    return rmidx, fout


def CPL_B(X, M):
    """
    Input:
        X: points feature [B, N, C]
        M: down sampling num
    Return:
        F: down sampled points feature, [B, M, C]
    """
    device = X.device
    B, N, C = X.size()
    X = X.detach().cpu().numpy()
    F = np.zeros((B, M, C), dtype=np.float32)
    IDX = np.zeros((B, M))
    for b in range(B):
        IDX[b], F[b] = CPL(X[b], M)
    IDX = torch.from_numpy(IDX).to(device)
    F = torch.from_numpy(F).to(device)
    return IDX, F


# Down sampling: gumbel sampling
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    device = logits.device
    y = logits + sample_gumbel(logits.size()).to(device)
    return torch.nn.functional.softmax(y / temperature, dim=-1).to(device)


def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    device = logits.device
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1]).to(device)
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


class Logger():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


if __name__ == '__main__':
    points = torch.rand(32, 1024, 3).to('cuda')
    features = torch.rand(32, 1024, 128).to('cuda')
    new_points, new_features = sample_and_knn_group(512, 32, points, features)
    print(new_points.size())
    print(new_features.size())