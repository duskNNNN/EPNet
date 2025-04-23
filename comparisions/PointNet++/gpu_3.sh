#!/bin/bash
# AB(wait)
# CUDA_VISIBLE_DEVICES=4 python train_partseg_strategy.py \
# --model pointnet2_part_seg_msg_unet  \
# --gpu 4 \
# --optimizer AdamW \
# --log_dir AB_resize 

# AC(wait)
# CUDA_VISIBLE_DEVICES=7 python train_partseg_strategy.py \
# --model pointnet2_part_seg_msg_fe  \
# --gpu 7 \
# --optimizer AdamW \
# --log_dir AC_resize 

# AD(wait)
# CUDA_VISIBLE_DEVICES=6 python train_partseg_strategy.py \
# --model pointnet2_part_seg_msg_fdr  \
# --gpu 6 \
# --optimizer AdamW \
# --log_dir AD_resize

# BC(wait)
# CUDA_VISIBLE_DEVICES=3 python train_partseg.py \
# --model pointnet2_part_seg_msg_unet_fe  \
# --gpu 3 \
# --optimizer Adam \
# --log_dir BC

# ABC
# CUDA_VISIBLE_DEVICES=5 python train_partseg_strategy.py \
# --model pointnet2_part_seg_msg_unet_fe  \
# --gpu 5 \
# --optimizer AdamW \
# --log_dir ABC_resize 

# ABD
# CUDA_VISIBLE_DEVICES=3 python train_partseg_strategy.py \
# --model pointnet2_part_seg_msg_unet_fdr  \
# --gpu 3 \
# --optimizer AdamW \
# --log_dir ABD

# ABCD
# CUDA_VISIBLE_DEVICES=3 python train_partseg_strategy.py \
# --model pointnet2_part_seg_msg_unet_fe_fdr  \
# --gpu 4 \
# --optimizer AdamW \
# --log_dir ABCD 