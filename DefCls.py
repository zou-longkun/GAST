import numpy as np
import random
import torch
import utils.pc_utils as pc_utils


def defcls_input(X, norm_curv, lookup, device='cuda:0', NREGIONS=3):
    """
    Deform a region in the point cloud.
    Input:
        args - commmand line arguments
        X - Point cloud [B, C, N]
        norm_curv - norm and curvature [B, N, D]
        lookup - regions center point
        device - cuda/cpu
    Return:
        X - Point cloud with a deformed region
        def_label - {0,1,...,26} indicating the deform class (deform region location) respectively
    """

    # get points' regions 
    regions = pc_utils.assign_region_to_point(X, device, NREGIONS)  # [B, N] N:the number of region_id

    n = NREGIONS
    curv_conf = torch.ones(X.shape[0]).to(device)
    region_ids = np.random.permutation(n ** 3)
    region_ids = torch.from_numpy(region_ids).to(device)
    def_label = torch.zeros(regions.size(0)).long().to(device)  # binary mask of deformed points

    for b in range(X.shape[0]):
        for i in region_ids:
            ind = regions[b, :] == i  # [N]
            # if there are enough points in the region
            if torch.sum(ind) >= pc_utils.MIN_POINTS:
                region = lookup[i].cpu().numpy()  # current region average point
                def_label[b] = i
                num_points = int(torch.sum(ind).cpu().numpy())
                rnd_pts = pc_utils.draw_from_gaussian(region, num_points)  # generate region deformation points
                # rnd_ind = random.sample(range(0, X.shape[2]), num_points)
                # X[b, :, ind] = X[b, :, rnd_ind]
                curv_conf[b] = norm_curv[b, ind, -1].abs().sum() / norm_curv[b, :, -1].abs().sum()
                X[b, :3, ind] = torch.tensor(rnd_pts, dtype=torch.float).to(device)  # replace with region deformation points
                break  # move to the next shape in the batch

    return X, def_label, curv_conf


def calc_loss(args, logits, labels, curv_conf, criterion):
    """
    Calc. DefCls loss.
    Return: loss
    """
    prediction = logits['def_cls']
    curv_conf = curv_conf.reshape(-1, 1)
    # prediction_curv_conf = logits['curv_conf']
    # loss = args.DefCls_weight * criterion(prediction, labels) + torch.abs(curv_conf + prediction_curv_conf -1).mean()
    loss = criterion(prediction, labels) # * curv_conf * 100
    loss = args.DefCls_weight * loss.mean()
    return loss
