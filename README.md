# Self-Supervised Learning for Domain Adaptation on Point-Clouds

<p align="center"> 
    <img src="./resources/model.png" width="400">
</p> 
 
 ### Introduction
The point cloud representation of an object can have a large geometric variation in view of inconsistent data acquisition procedure, which thus leads to domain discrepancy due to diverse and uncontrollable shape representation cross datasets. To improve discrimination on unseen distribution of point-based geometries in a practical and feasible perspective, this paper proposes a new method of geometry-aware self-training (GAST) for unsupervised domain adaptation of object point cloud classification. Specifically, this paper aims to learn a domain-shared representation of semantic categories, via two novel self-supervised geometric learning tasks as feature regularization. On one hand, the representation learning is empowered by a linear mixup of point cloud samples with their self-generated rotation labels, to capture a global topological configuration of local geometries. On the other hand, a diverse point distribution across datasets can be normalized with a novel curvature-aware distortion localization. Experiments on the PointDA-10 dataset show that our GAST method can significantly outperform the state-of-the-art methods.

[[Paper]](https://arxiv.org/pdf/2108.09169.pdf)

### Instructions
Clone repo and install it
```bash
git clone https://github.com/zou-longkun/gast.git
cd gast
pip install -r requirements.txt
```

Download data:
```bash
cd ./data
python download.py
```

Run GAST on both source and target
```
python train_Norm.py 
```


### Citation
Please cite this paper if you want to use it in your work,
```
@article{achituve2020self,
  title={Self-Supervised Learning for Domain Adaptation on Point Clouds},
  author={Achituve, Idan and Maron, Haggai and Chechik, Gal},
  journal={arXiv preprint arXiv:2003.12641},
  year={2020}
}
```
 
### Shape Reconstruction
<p align="center"> 
    <img src="./resources/reconstruction.png">
</p> 
 
 
### Acknowledgement
Some of the code in this repoistory was taken (and modified according to needs) from the follwing sources:
[[PointNet]](https://github.com/charlesq34/pointnet), [[PointNet++]](https://github.com/charlesq34/pointnet2), [[DGCNN]](https://github.com/WangYueFt/dgcnn), [[PointDAN]](https://github.com/canqin001/PointDAN), [[Reconstructing_space]](http://papers.nips.cc/paper/9455-self-supervised-deep-learning-on-point-clouds-by-reconstructing-space), [[Mixup]](https://github.com/facebookresearch/mixup-cifar10),[[DefRec]](https://github.com/idanachi/DefRec_and_PCM.git)


