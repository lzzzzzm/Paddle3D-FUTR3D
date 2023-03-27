# Paddle3D复现FUTR3D

本项目基于Paddle3D套件，复现论文FUTR3D

## 数据集准备

```
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
```

将数据准备成如此格式，并将data文件夹放置根目录处，其中nuscenes_infos_train.pkl和nuscenes_infos_val.pkl文件，均为使用mmdet3d处理nuscenes的方法处理得到。参考如下：[mmdet3d处理连接](https://mmdetection3d.readthedocs.io/zh_CN/latest/datasets/nuscenes_det.html)

## 依赖安装

```
cd Paddle3D-FUTR3D
pip install -r requirements.txt --user
python setup.py install --user
```

## 前向推理

```
cd Paddle3D-FUTR3D
python tools/evaluate.py --config configs/futr3d/futr3d_cam_lidar.yml --model cam_lidar_pretrained.pdparams
```

## 训练代码

```
cd Paddle3D-FUTR3D
python tools/evaluate.py --config configs/futr3d/futr3d_cam_lidar.yml --model cam_lidar_init.pdparams
```