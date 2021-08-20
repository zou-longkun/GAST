import numpy as np
import torch
from DefCls import defcls_input
import loss.pc_utils as pc_utils
from plyfile import PlyData, PlyElement

lookup = torch.Tensor(pc_utils.region_mean(3))
file = './1_0097.ply'

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x, y, z in pc])
    return pc_array


def write_ply(save_path, points, text=True):
    """
    save_path : path to save: '/yy/XX.ply'
    pt: point_cloud: size (N,3)
    """
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(save_path)


pc = read_ply(file)
x = pc_utils.scale_to_unit_cube(pc)
x = np.expand_dims(x, 0)  # [1,2048,3]
x = x.transpose([0, 2, 1])  # [1,3,2048]
x = torch.tensor(x)
deform_pc, label, _ = defcls_input(x, lookup)
deform_pc = deform_pc.permute(0, 2, 1)
deform_pc = deform_pc.squeeze().detach().numpy()
write_ply('./distortion.ply', deform_pc)