import numpy as np
import os


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


def extract_feature_point(data):
    curv = data[:, -1]
    curv = abs(curv)
    ind = np.argsort(curv)
    ind = ind[:512]
    fea_points = data[ind]
    return fea_points


if __name__ == '__main__':
    root = "F:\\SSL_PDA\\data\\pointDA_data\\modelnet_norm_curv\\"
    save_root = "F:\\SSL_PDA\\data\\pointDA_data\\modelnet_feature_points\\"
    shapeName = getDir(root)
    shapeName.sort()
    PC = []
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
            # pc = data[:, :3]
            feature_points = extract_feature_point(data)
            PC.append(feature_points)
    assert (len(PC)==len(Shape))
    print(len(PC))
    print(len(Shape))
    print(len(File))

    for j in range(len(Shape)):
        s = Shape[j]
        f = File[j]
        pc = PC[j]
        # save_path = os.path.join(save_root, str(i))
        if not os.path.exists(os.path.join(save_root, s, 'train')):
            os.makedirs(os.path.join(save_root, s, 'train'))
        saveFile(pc, save_root, s, f, 'train')




