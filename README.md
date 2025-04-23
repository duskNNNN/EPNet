# EPNet: Efficient Part Segmentation for Dense Point Clouds
![](https://p.sda1.dev/21/714995104f8cab962bb744716edb211f/fig1.png)
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

## Visual

You should set `batch-size` to 1. After running, use `Cloudcompare` to view the generated files

**LVPC**

```shell
python test_partseg_save.py --cfg config/ShapeNetPart/test_LVPC_save.json
```

![](https://p.sda1.dev/21/3f1f51b0435e3df9de5cb69c0afb410c/fig6.png)

## Main Results

| Method      | Reference  |    OA↑    | mIoU↑      | VRAM↓      | Train↓    | Params↓   |                     Checkpoints Download                     | logs                                                         |
| :---------- | :--------- | :-------: | :--------- | :--------- | --------- | --------- | :----------------------------------------------------------: | ------------------------------------------------------------ |
| PointMLP    | ICLR 22    |   86.6%   | 53.64%     | 19.9G      | 14.7H     | 16.7M     | [PointMLP.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/PointMLP.pth) | [PointMLP.txt](https://github.com/duskNNNN/EPNet/blob/root/comparisions/og/part_seg/EPNet_LVPC/logs/EPNet.txt) |
| PointNeXt   | NeurIPS 22 |     -     | 34.46%     | 40.8G      | 59.7H     | 22.4M     | [PointNeXt.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/PointNeXt.pth) | [EPNet.txt](https://github.com/duskNNNN/EPNet/blob/root/log/part_seg/EPNet_LVPC/logs/EPNet.txt) |
| Point-BERT  | CVPR22     |   76.8%   | 40.86%     | 24.7G      | 17.9H     | 27.05M    | [Point-BERT.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/Point-BERT.pth) | [EPNet.txt](https://github.com/duskNNNN/EPNet/blob/root/log/part_seg/EPNet_LVPC/logs/EPNet.txt) |
| Point-MAE   | ECCV 22    |   93.8%   | 78.53%     | 26.1G      | 17.1H     | 27.05M    | [Point-MAE.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/Point-MAE.pth) | [EPNet.txt](https://github.com/duskNNNN/EPNet/blob/root/log/part_seg/EPNet_LVPC/logs/EPNet.txt) |
| Point-M2AE  | NeurIPS 22 |   93.6%   | 79.89%     | 46.2G      | 19.7H     | 25.47M    | [Point-M2AE.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/Point-M2AE.pth) | [EPNet.txt](https://github.com/duskNNNN/EPNet/blob/root/log/part_seg/EPNet_LVPC/logs/EPNet.txt) |
| ACT         | ICLR 23    |   93.6%   | 78.65%     | 42.5G      | 18.2H     | 27.05M    | [ACT.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/ACT.pth) | [EPNet.txt](https://github.com/duskNNNN/EPNet/blob/root/log/part_seg/EPNet_LVPC/logs/EPNet.txt) |
| PointGPT    | NeurIPS 23 |   91.3%   | 69.39%     | 42.9G      | 19.2H     | 24.69M    | [PointGPT.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/PointGPT.pth) | [EPNet.txt](https://github.com/duskNNNN/EPNet/blob/root/log/part_seg/EPNet_LVPC/logs/EPNet.txt) |
| ReCon       | ICML 23    |   93.8%   | 78.18%     | 33.0G      | 22.4H     | 48.53M    | [ReCon.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/ReCon.pth) | [EPNet.txt](https://github.com/duskNNNN/EPNet/blob/root/log/part_seg/EPNet_LVPC/logs/EPNet.txt) |
| ShapeLLM    | ECCV 24    |   90.1%   | 72.75%     | 30.6G      | 4.66H     | 48.53M    | [ShapeLLM.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/ShapeLLM.pth) | [EPNet.txt](https://github.com/duskNNNN/EPNet/blob/root/log/part_seg/EPNet_LVPC/logs/EPNet.txt) |
| PointMamba  | NeurIPS 24 |   92.8%   | 77.66%     | 23.4G      | 5.50H     | 5.78M     | [PointMamba.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/PointMamba.pth) | [EPNet.txt](https://github.com/duskNNNN/EPNet/blob/root/log/part_seg/EPNet_LVPC/logs/EPNet.txt) |
| PointRWKV   | AAAI 25    |   92.1%   | 75.29%     | 44.7G      | 5.16H     | 27.05M    | [PointRWKV.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/PointRWKV.pth) | [EPNet.txt](https://github.com/duskNNNN/EPNet/blob/root/log/part_seg/EPNet_LVPC/logs/EPNet.txt) |
| PointNet++  | NeurIPS 17 |   91.0%   | 70.76%     | 14.4G      | 4.16H     | **1.47M** | [PointNet++.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/PointNet++.pth) | [EPNet.txt](https://github.com/duskNNNN/EPNet/blob/root/log/part_seg/EPNet_LVPC/logs/EPNet.txt) |
| EPNet(Ours) | ICMR 25    | **92.7%** | **80.46%** | **10.64G** | **2.47H** | 3.90M     | [EPNet.pth](https://github.com/duskNNNN/EPNet/releases/download/camera-ready/EPNet.pth) | [EPNet.txt](https://github.com/duskNNNN/EPNet/blob/root/log/part_seg/EPNet_LVPC/logs/EPNet.txt) |




