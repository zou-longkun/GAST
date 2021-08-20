import numpy as np
import torch
import utils.pc_utils as pc_utils


def rotcls_input(X, device='cuda:0'):
    """
    Rotate point cloud.
    Input:
        args - commmand line arguments
        X - Point cloud [B, N, C]
        device - cuda/cpu
    Return:
        mixed_X - mixed rotated point cloud
        pos_label - {0,1,2,3} indicating the rotate-angle 0, 90, 180, 270 respectively
    """
    n = pc_utils.NROTATIONS
    batch_size, _, num_points = X.size()
    mixed_X = X.clone().cpu().numpy()  # [B, C, N]
    pos_label_a = pos_label_b = torch.zeros(batch_size).to(device)

    # draw lambda from beta distribution
    lam = np.random.beta(1, 1)
    num_pts_a = round(lam * num_points)
    num_pts_b = num_points - round(lam * num_points)
    pts_indices_a, pts_vals_a = pc_utils.farthest_point_sample_np(mixed_X, num_pts_a)
    pts_indices_b, pts_vals_b = pc_utils.farthest_point_sample_np(mixed_X, num_pts_b)
    pts_vals_a = np.transpose(pts_vals_a,(0,2,1))   # [B, N1, C]
    pts_vals_b = np.transpose(pts_vals_b,(0,2,1))   # [B, N2, C]
    for b in range(batch_size):
        pos_id_a = np.random.randint(n)
        pts_vals_a[b, :, :] = pc_utils.rotate_shape(pts_vals_a[b, :, :], "x", np.pi * pos_id_a/2)
        pos_label_a[b] = pos_id_a
        pos_id_b = np.random.randint(n)
        pts_vals_b[b, :, :] = pc_utils.rotate_shape(pts_vals_b[b, :, :], "y", np.pi * pos_id_b/2)
        pos_label_b[b] = pos_id_b

    mixed_X = np.concatenate((pts_vals_a, pts_vals_b), 1).astype('float32')  # convex combination [B, N, C]
    mixed_X = torch.from_numpy(mixed_X).to(device)
    mixed_X = mixed_X.transpose(1, 2)  # [B, C, N]
    points_perm = torch.randperm(num_points).to(device)  # draw random permutation of points in the shape
    mixed_X = mixed_X[:, :, points_perm]
    pos_label_a = pos_label_a.long().to(device)
    pos_label_b = pos_label_b.long().to(device)

    return mixed_X, (pos_label_a, pos_label_b, lam)


def calc_loss(args, logits, pos_vals, criterion):
    """
    Calc. RotCls loss.
    Return: loss
    """
    pos_label_a, pos_label_b, lam = pos_vals
    loss = lam * criterion(logits['rot_cls1'], pos_label_a) + (1 - lam) * criterion(logits['rot_cls2'], pos_label_b)
    loss *= args.RotCls_weight
    return loss
