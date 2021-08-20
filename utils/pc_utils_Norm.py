import torch
import numpy as np
import random

eps = 10e-4
eps2 = 10e-6
KL_SCALER = 10.0
MIN_POINTS = 20
RADIUS = 0.5
NREGIONS = 3
NROTATIONS = 4
N = 16
K = 4
NUM_FEATURES = K * 3 + 1


def region_mean(num_regions):
    """
    Input:
        num_regions - number of regions
    Return:
        means of regions 
    """
    
    n = num_regions
    lookup = []
    d = 2 / n  # the cube size length
    #  construct all possibilities on the line [-1, 1] in the 3 axes
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            for k in range(n-1, -1, -1):
                lookup.append([1 - d * (i + 0.5), 1 - d * (j + 0.5), 1 - d * (k + 0.5)])
    lookup = np.array(lookup)  # n**3 x 3
    return lookup


def assign_region_to_point(X, device='cuda:0', NREGIONS=3):
    """
    Input:
        X: point cloud [B, C, N]
        device: cuda:0, cpu
    Return:
        Y: Region assignment per point [B, N]
    """

    n = NREGIONS
    d = 2 / n
    X_clip = torch.clamp(X, -0.99999999, 0.99999999)  # [B, C, N]
    batch_size, _, num_points = X.shape
    Y = torch.zeros((batch_size, num_points), device=device, dtype=torch.long)  # label matrix  [B, N]

    # The code below partitions all points in the shape to voxels.
    # At each iteration find per axis the lower threshold and the upper threshold values
    # of the range according to n (e.g., if n=3, then: -1, -1/3, 1/3, 1 - there are 3 ranges)
    # and save points in the corresponding voxel if they fall in the examined range for all axis.
    region_id = 0
    for x in range(n):
        for y in range(n):
            for z in range(n):
                # lt= lower threshold, ut = upper threshold
                x_axis_lt = -1 + x * d < X_clip[:, 0, :]        # [B, 1, N]
                x_axis_ut = X_clip[:, 0, :] < -1 + (x + 1) * d  # [B, 1, N]
                y_axis_lt = -1 + y * d < X_clip[:, 1, :]        # [B, 1, N]
                y_axis_ut = X_clip[:, 1, :] < -1 + (y + 1) * d  # [B, 1, N]
                z_axis_lt = -1 + z * d < X_clip[:, 2, :]        # [B, 1, N]
                z_axis_ut = X_clip[:, 2, :] < -1 + (z + 1) * d  # [B, 1, N]
                # get a mask indicating for each coordinate of each point of each shape whether
                # it falls inside the current inspected ranges
                in_range = torch.cat([x_axis_lt, x_axis_ut, y_axis_lt, y_axis_ut,
                                      z_axis_lt, z_axis_ut], dim=1).view(batch_size, 6, -1)  # [B, 6, N]
                # per each point decide if it falls in the current region only if in all
                # ranges the value is 1 (i.e., it falls inside all the inspected ranges)
                mask, _ = torch.min(in_range, dim=1)  # [B, N]
                Y[mask] = region_id  # label each point with the region id
                region_id += 1

    return Y


def collapse_to_point(x, device):
    """
    Input:
        X: point cloud [C, N]
        device: cuda:0, cpu
    Return:
        x: A deformed point cloud. Randomly sample a point and cluster all point
        within a radius of RADIUS around it with some Gaussian noise.
        indices: the points that were clustered around x
    """
    # get pairwise distances
    inner = -2 * torch.matmul(x.transpose(1, 0), x)
    xx = torch.sum(x ** 2, dim=0, keepdim=True)
    pairwise_distance = xx + inner + xx.transpose(1, 0)

    # get mask of points in threshold
    mask = pairwise_distance.clone()
    mask[mask > RADIUS ** 2] = 100
    mask[mask <= RADIUS ** 2] = 1
    mask[mask == 100] = 0

    # Choose only from points that have more than MIN_POINTS within a RADIUS of them
    pts_pass = torch.sum(mask, dim=1)
    pts_pass[pts_pass < MIN_POINTS] = 0
    pts_pass[pts_pass >= MIN_POINTS] = 1
    indices = (pts_pass != 0).nonzero()

    # pick a point from the ones that passed the threshold
    point_ind = np.random.choice(indices.squeeze().cpu().numpy())
    point = x[:, point_ind]  # get point
    point_mask = mask[point_ind, :]  # get point mask

    # draw a gaussian centered at the point for points falling in the region
    indices = (point_mask != 0).nonzero().squeeze()
    x[:, indices] = torch.tensor(draw_from_gaussian(point.cpu().numpy(), len(indices)), dtype=torch.float).to(device)
    return x, indices


def draw_from_gaussian(mean, num_points):
    """
    Input:
        mean: a numpy vector
        num_points: number of points to sample
    Return:
        points sampled around the mean with small std
    """
    return np.random.multivariate_normal(mean, np.eye(3) * 0.1, num_points).T  # 0.001


def draw_from_uniform(gap, region_mean, num_points):
    """
    Input:
        gap: a numpy vector of region x,y,z length in each direction from the mean
        region_mean:
        num_points: number of points to sample
    Return:
        points sampled uniformly in the region
    """
    return np.random.uniform(region_mean - gap, region_mean + gap, (num_points, 3)).T


def farthest_point_sample(xyz, npoint, device='cuda:0'):
    """
    Input:
        xyz: pointcloud data, [B, C, N]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    B, C, N = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  # B x npoint
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    centroids_vals = torch.zeros(B, C, npoint).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest  # save current chosen point index
        centroid = xyz[batch_indices, :, farthest].view(B, C, 1)  # get the current chosen point value
        centroids_vals[:, :, i] = centroid[:, :, 0].clone()
        dist = torch.sum((xyz - centroid) ** 2, 1)  # euclidean distance of points from the current centroid
        mask = dist < distance  # save index of all point that are closer than the current max distance
        distance[mask] = dist[mask]  # save the minimal distance of each point from all points that were chosen until now
        farthest = torch.max(distance, -1)[1]  # get the index of the point farthest away
    return centroids, centroids_vals


def farthest_point_sample_np(xyz, norm_curv, npoint):
    """
    Input:
        xyz: pointcloud data, [B, C, N]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    B, C, N = xyz.shape
    centroids = np.zeros((B, npoint), dtype=np.int64)
    distance = np.ones((B, N)) * 1e10
    farthest = np.random.randint(0, N, (B,), dtype=np.int64)
    batch_indices = np.arange(B, dtype=np.int64)
    centroids_vals = np.zeros((B, C, npoint))
    centroids_norm_curv_vals = np.zeros((B, 4, npoint))
    for i in range(npoint):
        centroids[:, i] = farthest  # save current chosen point index
        centroid = xyz[batch_indices, :, farthest].reshape(B, C, 1)  # get the current chosen point value
        centroid_norm_curv = norm_curv[batch_indices, :, farthest].reshape(B, 4, 1)
        centroids_vals[:, :, i] = centroid[:, :, 0].copy()
        centroids_norm_curv_vals[:, :, i] = centroid_norm_curv[:, :, 0].copy()
        dist = np.sum((xyz - centroid) ** 2, 1)  # euclidean distance of points from the current centroid
        mask = dist < distance  # save index of all point that are closer than the current max distance
        distance[mask] = dist[mask]  # save the minimal distance of each point from all points that were chosen until now
        farthest = np.argmax(distance, axis=1)  # get the index of the point farthest away
    return centroids, centroids_vals, centroids_norm_curv_vals


def rotate_shape(x, axis, angle):
    """
    Input:
        x: pointcloud data, [B, C, N]
        axis: axis to do rotation about
        angle: rotation angle
    Return:
        A rotated shape
    """
    R_x = np.asarray([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    R_y = np.asarray([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    R_z = np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    if axis == "x":
        return x.dot(R_x).astype('float32')
    elif axis == "y":
        return x.dot(R_y).astype('float32')
    else:
        return x.dot(R_z).astype('float32')


def random_rotate_one_axis(X, axis):
    """
    Apply random rotation about one axis
    Input:
        x: pointcloud data, [B, C, N]
        axis: axis to do rotation about
    Return:
        A rotated shape
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    if axis == 'x':
        R_x = [[1, 0, 0], [0, cosval, -sinval], [0, sinval, cosval]]
        X = np.matmul(X, R_x)
    elif axis == 'y':
        R_y = [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        X = np.matmul(X, R_y)
    else:
        R_z = [[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]]
        X = np.matmul(X, R_z)
    return X.astype('float32')


def translate_pointcloud(pointcloud):
    """
    Input:
        pointcloud: pointcloud data, [B, C, N]
    Return:
        A translated shape
    """
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    """
    Input:
        pointcloud: pointcloud data, [B, C, N]
        sigma:
        clip:
    Return:
        A jittered shape
    """
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud.astype('float32')


def scale_to_unit_cube(x):
    """
   Input:
       x: pointcloud data, [B, C, N]
   Return:
       A point cloud scaled to unit cube
   """
    if len(x) == 0:
        return x

    centroid = np.mean(x, axis=0)
    x -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(x) ** 2, axis=-1)))
    x /= furthest_distance
    return x


def dropout_points(x, norm_curv, num_points):
    """
    Randomly dropout num_points, and randomly duplicate num_points
   Input:
       x: pointcloud data, [B, C, N]
   Return:
       A point cloud dropouted num_points
   """
    ind = random.sample(range(0, x.shape[1]), num_points)
    ind_dpl = random.sample(range(0, x.shape[1]), num_points)
    x[:, ind, :] = x[:, ind_dpl, :]
    norm_curv[:, ind, :] = norm_curv[:, ind_dpl, :]
    return x, norm_curv


def remove_region_points(x, norm_curv, device):
    """
        Remove all points of a randomly selected region in the point cloud.
        Input:
            X - Point cloud [B, N, C]
            norm_curv: norm and curvature, [B, N, C]
        Return:
            X - Point cloud where points in a certain region are removed
        """
    # get points' regions
    regions = assign_region_to_point(x, device)  # [B, N] N:the number of region_id
    n = NREGIONS
    region_ids = np.random.permutation(n ** 3)
    for b in range(x.shape[0]):
        for i in region_ids:
            ind = regions[b, :] == i  # [N]
            # if there are enough points in the region
            if torch.sum(ind) >= 50:
                num_points = int(torch.sum(ind))
                rnd_ind = random.sample(range(0, x.shape[1]), num_points)
                x[b, ind, :] = x[b, rnd_ind, :]
                norm_curv[b, ind, :] = norm_curv[b, rnd_ind, :]
                break  # move to the next shape in the batch
    return x, norm_curv


def extract_feature_points(x, norm_curv, num_points, device="cuda:0"):
    """
   Input:
       x: pointcloud data, [B, N, C]
       norm_curv: norm and curvature, [B, N, C]
   Return:
       Feature points, [B, num_points, C]
   """
    IND = torch.zeros([x.size(0), num_points]).to(device)
    fea_pc = torch.zeros([x.size(0), num_points, x.size(2)]).to(device)
    for b in range(x.size(0)):
        curv = norm_curv[b, :, -1]
        curv = abs(curv)
        ind = torch.argsort(curv)
        ind = ind[:num_points]
        IND[b] = ind
        fea_pc[b] = x[b, ind, :]
    return fea_pc


def pc2voxel(x):
    # Args:
    #     x: size n x F where n is the number of points and F is feature size
    # Returns:
    #     voxel: N x N x N x (K x 3 + 1)
    #     index: N x N x N x K
    num_points = x.shape[0]
    data = np.zeros((N, N, N, NUM_FEATURES), dtype=np.float32)
    index = np.zeros((N, N, N, K), dtype=np.float32)
    x /= 1.05
    idx = np.floor((x + 1.0) / 2.0 * N)
    L = [[] for _ in range(N * N * N)]
    for p in range(num_points):
        k = int(idx[p, 0] * N * N + idx[p, 1] * N + idx[p, 2])
        L[k].append(p)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                u = int(i * N * N + j * N + k)
                if not L[u]:
                    data[i, j, k, :] = np.zeros((NUM_FEATURES), dtype=np.float32)
                elif len(L[u]) >= K:
                    choice = np.random.choice(L[u], size=K, replace=False)
                    local_points = x[choice, :] - np.array(
                        [-1.0 + (i + 0.5) * 2.0 / N, -1.0 + (j + 0.5) * 2.0 / N,
                         -1.0 + (k + 0.5) * 2.0 / N], dtype=np.float32)
                    data[i, j, k, 0: K * 3] = np.reshape(local_points, (K * 3))
                    data[i, j, k, K * 3] = 1.0
                    index[i, j, k, :] = choice
                else:
                    choice = np.random.choice(L[u], size=K, replace=True)
                    local_points = x[choice, :] - np.array(
                        [-1.0 + (i + 0.5) * 2.0 / N, -1.0 + (j + 0.5) * 2.0 / N,
                         -1.0 + (k + 0.5) * 2.0 / N], dtype=np.float32)
                    data[i, j, k, 0: K * 3] = np.reshape(local_points, (K * 3))
                    data[i, j, k, K * 3] = 1.0
                    index[i, j, k, :] = choice
    return data, index


def pc2voxel_B(x):
    """
   Input:
       x: pointcloud data, [B, num_points, C]
   Return:
       voxel: N x N x N x (K x 3 + 1)
       index: N x N x N x K
   """
    batch_size = x.shape[0]
    Data = np.zeros((batch_size, N, N, N, NUM_FEATURES), dtype=np.float32)
    Index = np.zeros((batch_size, N, N, N, K), dtype=np.float32)
    x = scale_to_unit_cube(x)
    for b in range(batch_size):
        pc = x[b]
        data, index = pc2voxel(pc)
        Data[b] = data
        Index[b] = index
    return Data, Index


def pc2image(X, axis, RESOLUTION=32):
    """
    Input:
        X: point cloud [N, C]
        axis: axis to do projection about
    Return:
        Y: image projected by 'X' along 'axis'. [32, 32]
    """

    n = RESOLUTION
    d = 2 / n
    X_clip = np.clip(X, -0.99999999, 0.99999999)  # [N, C]
    Y = np.zeros((n, n), dtype=np.float32)  # label matrix  [n, n]
    if axis == 'x':
        for y in range(n):
            for z in range(n):
                # lt= lower threshold, ut = upper threshold
                y_axis_lt = -1 + y * d < X_clip[:, 1]        # [N]
                y_axis_ut = X_clip[:, 1] < -1 + (y + 1) * d  # [N]
                z_axis_lt = -1 + z * d < X_clip[:, 2]        # [N]
                z_axis_ut = X_clip[:, 2] < -1 + (z + 1) * d  # [N]
                # get a mask indicating for each coordinate of each point of each shape whether
                # it falls inside the current inspected ranges
                in_range = np.concatenate([y_axis_lt, y_axis_ut, z_axis_lt, z_axis_ut], 0).reshape(4, -1)  # [4, N]
                # per each point decide if it falls in the current region only if in all
                # ranges the value is 1 (i.e., it falls inside all the inspected ranges)
                mask = np.min(in_range, 0)  # [N]: [False, ..., True, ...]
                if np.sum(mask) == 0:
                    continue
                Y[y, z] = (X_clip[mask, 0] + 1).mean()
    if axis == 'y':
        for x in range(n):
            for z in range(n):
                # lt= lower threshold, ut = upper threshold
                x_axis_lt = -1 + x * d < X_clip[:, 0]        # [N]
                x_axis_ut = X_clip[:, 0] < -1 + (x + 1) * d  # [N]
                z_axis_lt = -1 + z * d < X_clip[:, 2]        # [N]
                z_axis_ut = X_clip[:, 2] < -1 + (z + 1) * d  # [N]
                # get a mask indicating for each coordinate of each point of each shape whether
                # it falls inside the current inspected ranges
                in_range = np.concatenate([x_axis_lt, x_axis_ut, z_axis_lt, z_axis_ut], 0).reshape(4, -1)  # [4, N]
                # per each point decide if it falls in the current region only if in all
                # ranges the value is 1 (i.e., it falls inside all the inspected ranges)
                mask = np.min(in_range, 0)  # [N]
                if np.sum(mask) == 0:
                    continue
                Y[x, z] = (X_clip[mask, 1] + 1).mean()
    if axis == 'z':
        for x in range(n):
            for y in range(n):
                # lt= lower threshold, ut = upper threshold
                x_axis_lt = -1 + x * d < X_clip[:, 0]        # [N]
                x_axis_ut = X_clip[:, 0] < -1 + (x + 1) * d  # [N]
                y_axis_lt = -1 + y * d < X_clip[:, 1]        # [N]
                y_axis_ut = X_clip[:, 1] < -1 + (y + 1) * d  # [N]
                # get a mask indicating for each coordinate of each point of each shape whether
                # it falls inside the current inspected ranges
                in_range = np.concatenate([x_axis_lt, x_axis_ut, y_axis_lt, y_axis_ut], 0).reshape(4, -1)  # [4, N]
                # per each point decide if it falls in the current region only if in all
                # ranges the value is 1 (i.e., it falls inside all the inspected ranges)
                mask = np.min(in_range, 0)  # [N]
                if np.sum(mask) == 0:
                    continue
                Y[x, y] = (X_clip[mask, 2] + 1).mean()

    return Y


def pc2image_B(X, axis, device='cuda:0', RESOLUTION=32):
    """
    Input:
        X: point cloud [B, C, N]
        axis: axis to do projection about
    Return:
        Y: image projected by 'X' along 'axis'. [B, 32, 32]
    """
    n = RESOLUTION
    B = X.size(0)
    X = X.permute(0, 2, 1)  # [B, N, C]
    X = X.cpu().numpy()
    Y = np.zeros((B, n, n), dtype=np.float32)  # label matrix  [B, n, n]
    for b in range(B):
        Y[b] = pc2image(X[b], axis, n)
    Y = torch.from_numpy(Y).to(device)
    return Y


# Down sampling: critical point layer
def CPL(x, ratio):
    """
    Input:
        x: points feature [N, C]
        ratio: down sampling ratio
    Return:
        f_out: down sampled points feature, [M, C]
    """
    num_sample = int(np.size(x, 0) / ratio)
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
    rmidx = np.resize(midx, num_sample)
    fout = x[rmidx]
    return fout


def CPL_B(X, ratio, device='cuda:0',):
    """
    Input:
        X: points feature [B, C, N]
        ratio: down sampling ratio
    Return:
        F: down sampled points feature, [B, C, M]
    """
    B, C, N = X.size()
    M = int(N / ratio)
    X = X.permute(0, 2, 1)  # [B, N, C]
    X = X.cpu().numpy()
    F = np.zeros((B, M, C), dtype=np.float32)
    for b in range(B):
        F[b] = CPL(X[b], ratio)
    F = torch.from_numpy(F).to(device)
    F = F.permute(0, 2, 1)
    return F


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return torch.nn.functional.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
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


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
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


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
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


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


if __name__ == '__main__':
    lookup = region_mean(3)
    print(lookup.shape)
    x = np.random.rand(2, 3, 6)  # [B, C, N]
    print(x)
    x = scale_to_unit_cube(x)
    x = torch.from_numpy(x)
    print(x)
    #dropout_points(x, 2)
    y = pc2image_B(x, "x", RESOLUTION=6)
    print(y.shape)
    x = torch.stack((pc2image_B(x, "x", RESOLUTION=6), pc2image_B(x, "y", RESOLUTION=6), pc2image_B(x, "z", RESOLUTION=6)), dim=3)
    print(x)

