import numpy as np


# Function to create point cloud file
def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
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
\n
'''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)


if __name__ == '__main__':
    # Define name for output file
    output_file = 'bed_0001.ply'
    a = np.load("bed_0001.npy")
    print(a.shape)
    x = a[:, :3]
    print(a[1][6])
    b = np.float32(x)
    one = np.zeros((2048, 3))
    one[:, 0] = a[:, -1]
    one = np.float32(abs(one)) * 10
    #    points_3D = np.array([[1,2,3],[3,4,5]]) # 得到的3D点（x，y，z），即2个空间点
    #    colors = np.array([[0, 255, 255], [0, 255, 255]])   #给每个点添加rgb
    # Generate point cloud
    print("\n Creating the output file...\n")
    #    create_output(points_3D, colors, output_file)
    create_output(b, one, output_file)
