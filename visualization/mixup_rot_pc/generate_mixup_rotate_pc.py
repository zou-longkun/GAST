import numpy as np
import torch
import loss.pc_utils as pc_utils
import os
from plyfile import PlyData
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def posreg_input(X, device='cuda:0'):
    """
    Rotate point cloud.
    Input:
        args - commmand line arguments
        X - Point cloud [B, N, C]
        device - cuda/cpu
    Return:
        X - rotated point cloud
        pos_label - {0,1,2,3} indicating the rotate-angle 0, 90, 180, 270 respectively
    """
    n = pc_utils.NROTATIONS
    batch_size, _, num_points = X.size()
    mixed_X = X.clone().cpu().numpy()  # [B, C, N]

    # draw lambda from beta distribution
    #lam = np.random.beta(1, 1)
    lam = 0.5
    num_pts_a = round(lam * num_points)
    num_pts_b = num_points - round(lam * num_points)
    pts_indices_a, pts_vals_a = pc_utils.farthest_point_sample_np(mixed_X, num_pts_a)
    pts_indices_b, pts_vals_b = pc_utils.farthest_point_sample_np(mixed_X, num_pts_b)
    pts_vals_a = np.transpose(pts_vals_a,(0,2,1))   # [B, N1, C]
    pts_vals_b = np.transpose(pts_vals_b,(0,2,1))   # [B, N2, C]
    for b in range(batch_size):
        # pos_id_a = np.random.randint(n)
        pos_id_a = 0
        pts_vals_a[b, :, :] = pc_utils.rotate_shape(pts_vals_a[b, :, :], "x", np.pi * pos_id_a/2)
        pos_id_b = 2
        pts_vals_b[b, :, :] = pc_utils.rotate_shape(pts_vals_b[b, :, :], "Z", np.pi * pos_id_b/2)

    return pts_vals_a, pts_vals_b


def calc_loss(args, logits, pos_vals, criterion):
    """
    Calc. PosReg loss.
    Return: loss 
    """
    pos_label_a, pos_label_b, lam = pos_vals
    loss = lam * criterion(logits['pos_cls1'], pos_label_a) + (1 - lam) * criterion(logits['pos_cls2'], pos_label_b)
    loss *= args.PosReg_weight
    return loss

# Function to create point cloud file
def create_output(vertices, filename):
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')  # 必须先写入，然后利用write()在头部插入ply header
    ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)


if __name__ == '__main__':
    output_file = 'bed_0001.ply'
    X = np.load("bed_0001.npy")  # [2048, 3]

    # plydata = PlyData.read('9_0016.ply')
    # x_l = plydata['vertex']['x']
    # y_l = plydata['vertex']['y']
    # z_l = plydata['vertex']['z']
    # X = np.vstack((x_l, y_l, z_l))  # [3, 2048]  np.row_stack
    # X = X.transpose(1, 0)

    # X = np.dstack((x_l, y_l, z_l))
    # X = torch.from_numpy(X).transpose(1, 2)

    # X = np.column_stack((x_l, y_l, z_l))

    print(X.shape)

    X = torch.from_numpy(X).unsqueeze(0).transpose(1, 2)
    x1, x2 = posreg_input(X, device='cuda:0')
    x1 = x1.squeeze()
    x2 = x2.squeeze()
    c1 = np.ones((1024, 3))
    c1[:, 0] = 255
    c1 = np.float32(c1)
    c2 = np.ones((1024, 3))
    c2[:, 1] = 255
    c2 = np.float32(c2)
    v1 = np.hstack([x1.reshape(-1, 3), c1])
    v2 = np.hstack([x2.reshape(-1, 3), c2])
    mixed_X = np.concatenate((v1, v2), 0).astype('float32')
    points_perm = np.random.randint(0, 2048, size=2048)
    mixed_X = mixed_X[points_perm, :]
    #    points_3D = np.array([[1,2,3],[3,4,5]]) # 得到的3D点（x，y，z），即2个空间点
    #    colors = np.array([[0, 255, 255], [0, 255, 255]])   #给每个点添加rgb
    # Generate point cloud
    print("\n Creating the output file...\n")
    #    create_output(points_3D, colors, output_file)
    create_output(mixed_X, output_file)

