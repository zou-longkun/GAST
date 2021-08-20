import numpy as np
import os
import h5py

# a = np.array([1,2,3,4])
path = "F:\\SSL_PDA\\data\\pointDA_data\\modelnet_norm_curv\\bathtub\\train\\"
path = "F:\\SSL_PDA\\data\\pointDA_data\\"
# np.save(path, a)
b = np.load(path + '000298.npy')
print(b.shape)


h5_name = "F:\\SSL_PDA\\data\\pointDA_data\\scannet_norm_curv\\train_1.h5"
f = h5py.File(h5_name, 'r')
data = f['data'][:]
label = f['label'][:]
f.close()
print(data[1][:, 3:])
print(label.shape)
