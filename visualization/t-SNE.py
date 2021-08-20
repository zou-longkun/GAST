import numpy as np
import json
from sklearn.manifold import TSNE

# Random state.
RS = 20150101

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=2, s=20, marker='x', c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')


emb_filename = "F:\\SSL_PDA\\embed_data\\"
# data_name = ['embed_data_s2m', 'embed_data_r2m', 'embed_data_unsu_s2m', 'embed_data_unsu_r2m']
data_name = ['embed_data']
for name in data_name:
     file = emb_filename + name
     data = json.load(open(file, 'r'))
     X = np.array(data[0])
     Y = np.array(data[1])
     proj = TSNE(random_state=RS).fit_transform(X)
     scatter(proj, Y)
     plt.savefig(name + '.png', dpi=300)
     plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# import json
# from sklearn import manifold
#
# def draw_tsne(emb_filename):
#     data = json.load(open(emb_filename, 'r'))
#     X = np.array(data[0])
#     y = np.array(data[1])
#     '''t-SNE'''
#     tsne = manifold.TSNE(n_components=2, init='pca', random_state=20150101)
#     X_tsne = tsne.fit_transform(X)
#     print(X.shape)
#     print(X_tsne.shape)
#     print(y.shape)
#     print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
#
#     '''嵌入空间可视化'''
#     x_min, x_max = X_tsne.min(0), X_tsne.max(0)
#     X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
#     plt.figure(figsize=(8, 8))
#     for i in range(X_norm.shape[0]):
#         plt.text(X_norm[i, 0], X_norm[i, 1], '*', color=plt.cm.Set1(y[i]),
#                  fontdict={'weight': 'bold', 'size': 10})
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()
#
# emb_filename = "F:\\SSL_PDA\\embed_data\\"
# data_name = ['embed_data_s2m', 'embed_data_r2m', 'embed_data_unsu_s2m', 'embed_data_unsu_r2m']
# for name in data_name:
#     file = emb_filename + name
#     draw_tsne(file)

