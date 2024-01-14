# [Uni-Fusion: Universal Continuous Mapping](https://jarrome.github.io/Uni-Fusion/)

[Yijun Yuan](https://jarrome.github.io/), [Andreas Nüchter](https://www.informatik.uni-wuerzburg.de/robotics/team/nuechter/)

[Preprint](https://arxiv.org/abs/2303.12678) |  [website](https://jarrome.github.io/Uni-Fusion/)

<p align="">
      <img src="assets/encoder.png" align="" width="45%">
      <img src="assets/PLV.png" align="riht" width="37%">
</p>

*Universal encoder **no need data train** | Voxel grid for mapping*

## TODO:
- [x] Upload the uni-encoder src (Jan.3)
- [x] Upload the env script (Jan.4)
- [x] Upload the recon. application (By Jan.8)
- [x] Upload the used ORB-SLAM2 support (Jan.8)
- [x] Upload the azure process for RGB,D,IR (Jan.8)
- [ ] Upload the seman. application (By Jan.14)
- [ ] Upload the Custom context demo (By Jan.16)
- [ ] Toy example for fast essembling Uni-Fusion into custom project
- [ ] Our current new project has a better option, I plan to replace this ORB-SLAM2 with that option after complete that work.

Because my annual PhD. meeting is on Jan.12, I have to prepare it, the TODO deadline will be postponed for 2 days.  

## 0. Env setting and install
* Create env
```
conda create -n uni python=3.8
conda activate uni

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install ninja functorch==0.2.1 numba open3d opencv-python trimesh
```

* install package
```
git clone https://github.com/Jarrome/Uni-Fusion.git && cd Uni-Fusion
# install uni package
python setup.py install
# install cuda function, this may take several minutes, please use `top` or `ps` to check
python uni/ext/__init__.py
```

* train a uni encoder from nothing in 1 second
```
python uni/encoder/uni_encoder_v2.py
```


<details>
<summary> optionally, you can install the [ORB-SLAM2](https://github.com/Jarrome/Uni-Fusion-use-ORB-SLAM2) that we use for tracking</summary>
  
```
cd external
git clone https://github.com/Jarrome/Uni-Fusion-use-ORB-SLAM2
cd [this_folder]
# this_folder is the absolute path for the orbslam2
# Add ORB_SLAM2/lib to PYTHONPATH and LD_LIBRARY_PATH environment variables
# I suggest putting this in ~/.bashrc
export PYTHONPATH=$PYTHONPATH:[this_folder]/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:[this_folder]/lib

./build.sh && ./build_python.sh
```
</details>

## 1. Reconstruction Demo

### download replica data
```
source scripts/download_replica.sh
```

### run demo
```
python demo.py configs/replica/office0.yaml
```

## 2. Custom context Demo

## 3. Semantic Demo
```
# install requirements
pip install tensorflow
pip install git+https://github.com/openai/CLIP.git

# download openseg ckpt
gsutil cp -r gs://cloud-tpu-checkpoints/detection/projects/openseg/colab/exported_model ./external/openseg/


```

## 4. Self-captured data
### Azure capturing
We provide the script to extract RGB, D and IR from azure.mp4: [azure_process](https://github.com/Jarrome/azure_process)

---


## Citation
If you find this work interesting, please cite us:
```bibtex
@article{yuan2024uni,
  title={Uni-Fusion: Universal Continuous Mapping},
  author={Yuan, Yijun and N{\"u}chter, Andreas},
  journal={IEEE Transactions on Robotics},
  year={2024},
  publisher={IEEE}
}
```

## Acknowledgement
* This implementation is on top of [DI-Fusion](https://github.com/huangjh-pub/di-fusion).
* We also borrow some dataset code from [NICE-SLAM](https://github.com/cvg/nice-slam).
* We thank the detailed response of questions from Kejie Li, Björn Michele, Songyou Peng and Golnaz Ghiasi.
