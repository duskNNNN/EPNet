<div align="center">
<h1>EPNet: <br>
Efficient Part Segmentation for Dense Point Clouds</h1>

[Cheng Wang](https://github.com/duskNNNN) , Wulong Hu , Minqian Wang, Zhenbo Cheng ,Yuanming Zhang, Fei Gao<sup>*</sup>

Zhejiang University of Technology

(*) equal contribution
</div>



![](https://p.sda1.dev/21/714995104f8cab962bb744716edb211f/fig1.png)

## Abstract

The segmentation of dense point clouds from industrial LiDAR scans presents challenges in computational overhead and VRAM usage, hindering the development of automated fast measurement systems. To address this, we propose EPNet, an efficient model for part segmentation of dense point clouds. EPNet employs a U-Net-like architecture with skip connections to merge original and recovered features, enhancing local feature extraction via KNN and cosine similarity. Factorization-dimensionality-reduction module based on self-attention overcomes the limitations of trilinear interpolation in feature recovery, improving both local and global feature fusion. In experiments on the LVPC dataset of dense vehicle point clouds, EPNet outperforms models from the past three years, achieving a 1.7\% accuracy improvement and a 9.7\% increase in average Instance IoU compared to PointNet++. EPNet also achieves a single-file inference time of under 1 second while requiring minimal GPU VRAM resources, demonstrating its potential for real-world industrial high-precision fast automated measurements. 

## Architecture
![](https://p.sda1.dev/21/7dcd70f2a367e4f05aaafc5ca441ffa2/fig2.png)
## Install
```shell
conda env create -f py38.yaml
pip install -r requirements.txt
```
## Datasets
**LVPC Dataset:** Download the offical data from [here](https://pan.baidu.com/s/1CUJvMX2bt8kKthb717pr1Q?pwd=s8hs). Unzip the file under `data/PartSeg/LVPC/`.

 The directory structure should be

```
|LVPC/
├──04379243/
│  ├── 1wqxaf.txt
│  ├── .......
│──train_test_split/
│──synsetoffset2category.txt
```

**ShapeNetPart Dataset:** Download the offical data from [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip). Unzip the file under `data/PartSegshapenetcore_partanno_segmentation_benchmark_v0_normal/`. 

The directory structure should be

```
|shapenetcore_partanno_segmentation_benchmark_v0_normal/
├──02691156/
│  ├── 1a04e3eab45ca15dd86060f189eb133.txt
│  ├── .......
│── .......
│──train_test_split/
│──synsetoffset2category.txt
```

## Train

**LVPC**

```shell
python -m torch.distributed.launch --nproc_per_node=1 --master_port 29502  --use_env train_partseg_ddp.py --cfg config/ShapeNetPart/train_LVPC.json
```

**ShapeNetPart**

```shell
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502  --use_env train_partseg_ddp.py --cfg config/ShapeNetPart/train_Shapenetpart.json
```
## Test

**LVPC**

```shell
python test_partseg.py --cfg config/ShapeNetPart/test_LVPC.json
```

**ShapeNetPart**

```shell
python test_partseg.py --cfg config/ShapeNetPart/test_Shapenetpart.json
```

## Main Results

### Train

|   Method    |  Reference   |    OA↑    |   mIoU↑    |   VRAM↓    |  Train↓   |  Params↓  |                     Checkpoints Download                     |                             logs                             |
| :---------: | :----------: | :-------: | :--------: | :--------: | :-------: | :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  PointMLP   |  ICLR 2022   |   86.6%   |   53.64%   |   19.9G    |   14.7H   |   16.7M   | [PointMLP.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/PointMLP.pth) | [PointMLP.txt](https://github.com/duskNNNN/EPNet/blob/root/comparisions/PointMLP/part_segmentation/checkpoints/pointMLP_demo_v3/pointMLP_demo_v3_train.log) |
|  PointNeXt  | NeurIPS 2022 |     -     |   34.46%   |   40.8G    |   59.7H   |   22.4M   | [PointNeXt.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/PointNeXt.pth) | [PointNeXt.txt](https://github.com/duskNNNN/EPNet/blob/root/comparisions/PointNeXt/log/shapenetpart/shapenetpart-train-pointnext-s-ngpus1-seed171-20241022-100411-kXa9X7Gq2ssq8doZvszZwj/shapenetpart-train-pointnext-s-ngpus1-seed171-20241022-100411-kXa9X7Gq2ssq8doZvszZwj.log) |
| Point-BERT  |  CVPR 2022   |   76.8%   |   40.86%   |   24.7G    |   17.9H   |  27.05M   | [Point-BERT.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/Point-BERT.pth) | [Point-BERT.txt](https://github.com/duskNNNN/EPNet/blob/root/comparisions/Point-BERT/segmentation/log/part_seg/v3/logs/PointTransformer.txt) |
|  Point-MAE  |  ECCV 2022   |   93.8%   |   78.53%   |   26.1G    |   17.1H   |  27.05M   | [Point-MAE.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/Point-MAE.pth) | [Point-MAE.txt](https://github.com/duskNNNN/EPNet/blob/root/comparisions/Point-MAE/segmentation/log/part_seg/exp_v3/logs/pt.txt) |
| Point-M2AE  | NeurIPS 2022 |   93.6%   |   79.89%   |   46.2G    |   19.7H   |  25.47M   | [Point-M2AE.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/Point-M2AE.pth) | [Point-M2AE.txt](https://github.com/duskNNNN/EPNet/blob/root/comparisions/Point-M2AE/segmentation/log/part_seg/exp_v3/logs/Point_M2AE_SEG.txt) |
|     ACT     |  ICLR 2023   |   93.6%   |   78.65%   |   42.5G    |   18.2H   |  27.05M   | [ACT.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/ACT.pth) | [ACT.txt](https://github.com/duskNNNN/EPNet/blob/root/comparisions/ACT/part_segmentation/log/part_seg/exp_v3/logs/pt_20241106_123854.log) |
|  PointGPT   | NeurIPS 2023 |   91.3%   |   69.39%   |   42.9G    |   19.2H   |  24.69M   | [PointGPT.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/PointGPT.pth) | [PointGPT.txt](https://github.com/duskNNNN/EPNet/blob/root/comparisions/PointGPT/segmentation/log/part_seg/exp_v3_S/logs/pt.txt) |
|    ReCon    |  ICML 2023   |   93.8%   |   78.18%   |   33.0G    |   22.4H   |  48.53M   | [ReCon.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/ReCon.pth) | [ReCon.txt](https://github.com/duskNNNN/EPNet/blob/root/comparisions/ReCon/segmentation/log/part_seg/exp_v3/logs/pt.txt) |
|  ShapeLLM   |  ECCV 2024   |   90.1%   |   72.75%   |   30.6G    |   4.66H   |  48.53M   | [ShapeLLM.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/ShapeLLM.pth) | [ShapeLLM.txt](https://github.com/duskNNNN/EPNet/blob/root/comparisions/ShapeLLM/ReConV2/segmentation/log/part_seg/exp_v3_base/logs/pt.txt) |
| PointMamba  | NeurIPS 2024 |   92.8%   |   77.66%   |   23.4G    |   5.50H   |   5.78M   | [PointMamba.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/PointMamba.pth) | [PointMamba.txt](https://github.com/duskNNNN/EPNet/blob/root/comparisions/PointMamba/part_segmentation/log/part_seg/exp_v3/logs/pt_mamba.txt) |
|  PointRWKV  |  AAAI 2025   |   92.1%   |   75.29%   |   44.7G    |   5.16H   |  27.05M   | [PointRWKV.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/PointRWKV.pth) | [PointRWKV.txt](https://github.com/duskNNNN/EPNet/blob/root/comparisions/PointRWKV/segmentation/log/part_seg/LVPC/logs/pt.txt) |
| PointNet++  | NeurIPS 2017 |   91.0%   |   70.76%   |   14.4G    |   4.16H   | **1.47M** | [PointNet++.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/PointNet++.pth) | [PointNet++.txt](https://github.com/duskNNNN/EPNet/blob/root/comparisions/PointNet%2B%2B/log/part_seg/LVPC_origin/logs/pointnet2_part_seg_msg.txt) |
| EPNet(Ours) |  ICMR 2025   | **92.7%** | **80.46%** | **10.64G** | **2.47H** |   3.90M   | [EPNet.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/EPNet.pth) | [EPNet.txt](https://github.com/duskNNNN/EPNet/blob/root/log/part_seg/EPNet_LVPC/logs/EPNet.txt) |

### Test

|   Method    |  Reference   |    OA↑     |   mIoU↑    |   Time↓   |   VRAM↓   |                             logs                             |
| :---------: | :----------: | :--------: | :--------: | :-------: | :-------: | :----------------------------------------------------------: |
| PointNet++  | NeurIPS 2017 |   90.69%   |   70.78%   |   3.41s   |  14.49G   | [PointNet++.txt](https://github.com/duskNNNN/EPNet/blob/root/comparisions/PointNet%2B%2B/log/part_seg/LVPC_origin/eval.txt) |
|  Point-MAE  |  ECCV 2022   |   93.22%   |   77.84%   |   9.71s   |  25.26G   | [Point-MAE.txt](https://github.com/duskNNNN/EPNet/blob/root/comparisions/Point-MAE/segmentation/log/part_seg/exp_v3/eval.txt) |
| Point-M2AE  | NeurIPS 2022 |   92.48%   |   79.41%   |  10.11s   |  41.95G   | [Point-M2AE.txt](https://github.com/duskNNNN/EPNet/blob/root/comparisions/Point-M2AE/segmentation/log/part_seg/exp_v3/eval.txt) |
|     ACT     |  ICLR 2023   |   92.67%   |   78.39%   |   9.91s   |  25.26G   | [ACT.txt](https://github.com/duskNNNN/EPNet/blob/root/comparisions/ACT/part_segmentation/log/part_seg/exp_v3/eval.txt) |
|    ReCon    |  ICML 2023   |   92.93%   |   77.02%   |   9.57s   |  38.25G   | [ReCon.txt](https://github.com/duskNNNN/EPNet/blob/root/comparisions/ReCon/segmentation/log/part_seg/exp_v3/eval.txt) |
| PointMamba  | NeurIPS 2024 |   91.66%   |   76.13%   |   2.70s   |  30.21G   | [PointMamba.txt](https://github.com/duskNNNN/EPNet/blob/root/comparisions/PointMamba/part_segmentation/log/part_seg/exp_v3/eval.txt) |
| EPNet(Ours) |  ICMR 2025   | **93.01%** | **81.96%** | **0.47s** | **7.34G** | [EPNet.txt](https://github.com/duskNNNN/EPNet/blob/root/log/part_seg/EPNet_LVPC/eval.txt) |

### Visual

You should set `batch-size` to 1. After running, use `Cloudcompare` to view the generated files

**LVPC**

```shell
python test_partseg_save.py --cfg config/ShapeNetPart/test_LVPC_save.json
```

![](https://p.sda1.dev/21/3f1f51b0435e3df9de5cb69c0afb410c/fig6.png)