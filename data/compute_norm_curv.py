import numpy as np
import os
import open3d


def getDir(path):
    fList = os.listdir(path)
    F = []
    for i in fList:
        d = os.path.join(path,i)
        if os.path.isdir(d):
            F.append(i)
    return F


def getFilenames(path, shapeName, partition='train'):
    Dir = os.path.join(path, shapeName, partition)
    fList = os.listdir(Dir)
    files = []
    for i in fList:
        f = os.path.join(Dir, i)
        if os.path.isfile(f):
            files.append(i)
    return files


def scale_to_unit_cube(x):
    if len(x) == 0:
        return x
    centroid = np.mean(x, axis=0)
    x -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(x) ** 2, axis=-1)))
    x /= furthest_distance
    return x


def readFile(path, shapeName, fileName, partition='train'):
    path = os.path.join(path, shapeName, partition, fileName)
    pc = np.load(path)
    return pc


def saveFile(data, path, shapeName, fileName, partition='train'):
    path = os.path.join(path, shapeName, partition, fileName)
    np.save(path, data)


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


def compute_norm_included_angle(q_norm, norms, knn_indices=None):
    norm_included_angles = []
    norm_included_angle = 0
    if knn_indices is not None:
        for i in range(len(knn_indices)):
            cosine_value = np.dot(q_norm, norms[i]) / (np.linalg.norm(q_norm) * (np.linalg.norm(norms[i])))
            eps = 1e-6
            if 1.0 < cosine_value < 1.0 + eps:
                cosine_value = 1.0
            elif -1.0 - eps < cosine_value < -1.0:
                cosine_value = -1.0
            included_angle = np.arccos(cosine_value) * 180 / np.pi
            norm_included_angles.append(included_angle)
        norm_included_angle = np.average(norm_included_angles)
    return norm_included_angle


if __name__ == '__main__':
    root = "F:\\SSL_PDA\\data\\pointDA_data\\shapenet\\"
    save_root = "F:\\SSL_PDA\\data\\pointDA_data\\shapenet_norm_curv_angle\\"
    shapeName = getDir(root)
    shapeName.sort()
    PC = []
    Norm = []
    Curv = []
    Norm_included_angle = []
    Shape = []
    File = []
    for s in shapeName:
        print(s)
        files = getFilenames(root, s)
        files.sort() 
        for f in files:
            print(f)
            Shape.append(s)
            File.append(f)
            data = readFile(root, s, f)  # (2048, 3)
            pc = data[:, :3]
            # pc = scale_to_unit_cube(pc)
            point_cloud = open3d.geometry.PointCloud()
            point_cloud.points = open3d.utility.Vector3dVector(pc)  # open3d.Vector3dVector()
            kdtree = open3d.geometry.KDTreeFlann()
            kdtree.set_geometry(point_cloud)
            norms = []
            curvs = []
            norm_included_angles = []
            for j in range(pc.shape[0]):
                q = pc[j]
                q = np.float64(q)
                k, indices, dist = kdtree.search_knn_vector_3d(q, knn=32)
                indices = np.asarray(indices)
                # print(indices.shape) # (32,)
                norm, curv = compute_norm_and_curvature(pc, indices)
                norms.append(norm)
                curvs.append(curv)
            norms = np.array(norms)
            curvs = np.array(curvs).reshape(pc.shape[0], 1)
            # print("curvs.shape", curvs.shape)
            # print('norms.shape', norms.shape)
            for j in range(pc.shape[0]):
                q = pc[j]
                q_norm = norms[j]
                q = np.float64(q)
                k, indices, dist = kdtree.search_knn_vector_3d(q, knn=32)
                indices = np.asarray(indices)
                # print(indices.shape) # (32,)
                norm_included_angle = compute_norm_included_angle(q_norm, norms, indices)
                norm_included_angles.append(norm_included_angle)
            norm_included_angles = np.array(norm_included_angles).reshape(pc.shape[0], 1)
            # print(norm_included_angles.shape)
            PC.append(pc)
            Norm.append(norms)
            Curv.append(curvs)
            Norm_included_angle.append(norm_included_angles)
    assert (len(PC)==len(Shape))
    print(len(PC))
    print(len(Curv))
    print(len(Shape))
    print(len(File))

    Curv = np.array(Curv)
    cmin = np.min(Curv)
    cmax = np.max(Curv)
    Curv = 2*(Curv-cmin)/(cmax-cmin)-1
    for j in range(len(Shape)):
        s = Shape[j]
        f = File[j]
        curv = Curv[j].reshape(1024, 1)  # note that the point number of pointcloud in shapenet is 1024
        norm = Norm[j]
        norm_included_angle = Norm_included_angle[j]
        pc = PC[j]
        data = np.concatenate([pc, norm, curv, norm_included_angle], -1)
        # save_path = os.path.join(save_root, str(i))
        if not os.path.exists(os.path.join(save_root, s, 'train')):
            os.makedirs(os.path.join(save_root, s, 'train'))
        saveFile(data, save_root, s, f, 'train')




