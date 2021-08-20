import numpy as np
import os
import open3d
import h5py


def load_data_h5py_scannet10(h5_name):
    f = h5py.File(h5_name, 'r')
    data = f['data'][:]
    label = f['label'][:]
    f.close()
    return data, label


def scale_to_unit_cube(x):
    if len(x) == 0:
        return x
    centroid = np.mean(x, axis=0)
    x -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(x) ** 2, axis=-1)))
    x /= furthest_distance
    return x


def saveFile(data, path, h5_fileName, label):
    # path = os.path.join(path, h5_fileName)
    f = h5py.File(path + h5_fileName, 'w')
    f['data'] = data
    f['label'] = label
    f.close()


def compute_norm_and_curvature(pc, knn_indices=None):
    if knn_indices is not None:
        pc = pc[knn_indices]
    covariance = np.cov(pc.T)
    w, v = np.linalg.eig(covariance)
    v = v.T
    w = np.real(w)
    i = np.argmin(np.abs(w))
    norm = v[i]
    # assert (np.sum(np.abs(w)) != 0)
    if np.sum(np.abs(w)) == 0:
        curv = w[i] / (np.sum(np.abs(w)) + 1e-5)
    else:
        curv = w[i] / np.sum(np.abs(w))
    # assert curv is not complex
    return np.real(norm), np.real(curv)


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
    root = "F:\\SSL_PDA\\data\\pointDA_data\\scannet\\"
    save_root = "F:\\SSL_PDA\\data\\pointDA_data\\scannet_norm_curv_angle\\"
    fileName = ['train_0.h5', 'train_1.h5', 'train_2.h5']
    # fileName = ['test_0.h5']
    for f in fileName:
        print(f)
        PC = []
        Norm = []
        Curv = []
        Norm_included_angle = []
        H5_data = []
        h5_file = os.path.join(root, f)
        data, label = load_data_h5py_scannet10(h5_file)
        data = np.squeeze(data)
        for d in data:
            pc = d[:, :3]
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
                # print(indices.shape)
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
            Norm_included_angle.append(norm_included_angles)
            PC.append(pc)
            Norm.append(norms)
            Curv.append(curvs)

        print(len(Norm))
        print(len(Curv))
        Curv = np.array(Curv)
        cmin = np.min(Curv)
        cmax = np.max(Curv)
        Curv = 2*(Curv-cmin)/(cmax-cmin)-1
        print(Curv.shape)

        for j in range(len(Norm)):
            curv = Curv[j]
            norm = Norm[j]
            pc = PC[j]
            norm_included_angle = Norm_included_angle[j]
            pc = np.concatenate([pc, norm, curv, norm_included_angle], -1)
            H5_data.append(pc)

        if not os.path.exists(save_root):
            os.makedirs(save_root)
        saveFile(H5_data, save_root, f, label)




