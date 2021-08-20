import open3d
import numpy as np
import torch


def compute_norm_and_curvature(pc, knn_indices=None):
    if knn_indices is not None:
        pc = pc[knn_indices]
    covariance = np.cov(pc.T)
    w, v = np.linalg.eig(covariance)
    v = v.T
    w = np.real(w)
    i = np.argmin(np.abs(w))
    norm = v[i]
    curv = w[i] / np.sum(np.abs(w))
    # assert curv is not complex
    return norm, np.real(curv)


def normreg_input(X, device='cuda:0'):
    Norm = []
    Curv = []
    X = X.transpose(1, 2).cpu().numpy()
    for b in range(X.shape[0]):
        pc = X[b, :, :]
        point_cloud = open3d.geometry.PointCloud()
        point_cloud.points = open3d.utility.Vector3dVector(pc)
        kdtree = open3d.geometry.KDTreeFlann()
        kdtree.set_geometry(point_cloud)
        norms = []
        curvs = []
        for j in range(pc.shape[0]):
            q = pc[j]
            q = np.float64(q)
            k, indices, dist = kdtree.search_knn_vector_3d(q, knn=32)
            indices = np.asarray(indices)
            # print(indices.shape)
            norm, curv = compute_norm_and_curvature(pc, indices)
            norms.append(norm)
            curvs.append(curv)
        norms = np.array(norms)
        curvs = np.array(curvs).reshape(pc.shape[0], 1)
        Norm.append(norms)
        Curv.append(curvs)
    Norm = np.array(Norm)
    Curv = np.array(Curv)
    cmin = np.min(Curv)
    cmax = np.max(Curv)
    Curv = 2 * (Curv - cmin) / (cmax - cmin) - 1
    X = torch.from_numpy(X).to(device)
    X = X.transpose(1, 2)
    Norm = torch.from_numpy(Norm).to(device)
    Curv = torch.from_numpy(Curv).to(device)
    return X, Norm, Curv


def calc_loss(args, logits, labels):
    """
    Calc. PosReg loss.
    Return: loss
    """
    prediction = logits['norm_reg']  # [b, num_points, 4]
    loss = args.NormReg_weight * sum([torch.sum((y-x)**2)/prediction.shape[1] for x, y in zip(prediction, labels)])/len(labels)
    return loss




