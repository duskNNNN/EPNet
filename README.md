# EPNet: Efficient Part Segmentation for Dense Point Clouds
![](https://p.sda1.dev/21/714995104f8cab962bb744716edb211f/fig1.png)
## Architecture
![](https://p.sda1.dev/21/7dcd70f2a367e4f05aaafc5ca441ffa2/fig2.png)
## Install
```shell
conda env create -f py38.yaml
pip install -r requirements
```
## Datasets
Will be released later

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

## Comparison
![](https://p.sda1.dev/21/3f1f51b0435e3df9de5cb69c0afb410c/fig6.png)
