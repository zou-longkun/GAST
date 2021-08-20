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


def extract_feature_point(data):
    curv = data[:, -1]
    curv = abs(curv)
    ind = np.argsort(curv)
    ind = ind[:512]
    fea_points = data[ind]
    return fea_points


if __name__ == '__main__':
    root = "F:\\SSL_PDA\\data\\pointDA_data\\scannet_norm_curv\\"
    save_root = "F:\\SSL_PDA\\data\\pointDA_data\\scannet_feature_points\\"
    fileName = ['train_0.h5', 'train_1.h5', 'train_2.h5']
    # fileName = ['test_0.h5']
    for f in fileName:
        print(f)
        PC = []
        H5_data = []
        h5_file = os.path.join(root, f)
        data, label = load_data_h5py_scannet10(h5_file)
        data = np.squeeze(data)
        for d in data:
            pc = extract_feature_point(d)
            # pc = scale_to_unit_cube(pc)
            H5_data.append(pc)

        if not os.path.exists(save_root):
            os.makedirs(save_root)
        saveFile(H5_data, save_root, f, label)




