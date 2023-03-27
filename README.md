# Paddle3D复现FUTR3D

本项目基于Paddle3D套件，复现论文FUTR3D。在Aistudio上有对Mini数据集的代码使用：[AiStudio项目连接](https://aistudio.baidu.com/aistudio/projectdetail/5739205)


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
在Nuscenes-Mini上的推理结果如下
```
mAP: 0.5709                                                                     
mATE: 0.4564
mASE: 0.4526
mAOE: 0.4662
mAVE: 0.5146
mAAE: 0.2900
NDS: 0.5675
Eval time: 5.5s

Per-class results:
Object Class	AP	ATE	ASE	AOE	AVE	AAE
car	0.894	0.182	0.156	0.082	0.113	0.062
truck	0.794	0.205	0.171	0.034	0.066	0.000
bus	0.992	0.241	0.156	0.036	1.083	0.099
trailer	0.000	1.000	1.000	1.000	1.000	1.000
construction_vehicle	0.000	1.000	1.000	1.000	1.000	1.000
pedestrian	0.856	0.239	0.247	0.331	0.197	0.158
motorcycle	0.696	0.321	0.317	0.556	0.052	0.000
bicycle	0.612	0.199	0.191	0.156	0.605	0.000
traffic_cone	0.866	0.178	0.288	nan	nan	nan
barrier	0.000	1.000	1.000	1.000	nan	nan
```

## 训练代码

```
cd Paddle3D-FUTR3D
python tools/evaluate.py --config configs/futr3d/futr3d_cam_lidar.yml --model cam_lidar_init.pdparams
```

