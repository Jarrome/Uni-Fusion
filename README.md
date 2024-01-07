# [Uni-Fusion: Universal Continuous Mapping](https://jarrome.github.io/Uni-Fusion/)

[Yijun Yuan](https://jarrome.github.io/), [Andreas Nüchter](https://www.informatik.uni-wuerzburg.de/robotics/team/nuechter/)

[Preprint](https://arxiv.org/abs/2303.12678) |  [website](https://jarrome.github.io/Uni-Fusion/)


## TODO:
- [x] Upload the uni-encoder src (Jan.3)
- [x] Upload the env script (Jan.4)
- [x] Upload the recon. application (By Jan.8)
- [ ] Upload the seman. application (By Jan.12)
- [ ] Upload the used ORB-SLAM2 support
- [ ] Our current new project has a better option, I plan to replace this ORB-SLAM2 with that option after complete that work.

## 0. Env setting and install
* Create env
```
conda create -n uni python=3.8
conda activate uni

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install ninja functorch==0.2.1 numba open3d opencv-python trimesh
```

* install package
```
# install uni package
python setup.py install
# install cuda function, this may take several minutes, please use `top` or `ps` to check
python uni/ext/__init__.py
```

* build uni encoder in 1 second
```
python uni/encoder/uni_encoder_v2.py
```

## 1. Reconstruction Demo

* download replica data
```
source scripts/download_replica.sh
```

* run demo
```
python demo.py configs/replica/office0.yaml
```

