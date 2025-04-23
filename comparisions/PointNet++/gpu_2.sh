#!/bin/bash
# A(run)
# CUDA_VISIBLE_DEVICES=0 python train_partseg_strategy.py \
# --model pointnet2_part_seg_msg  \
# --gpu 0 \
# --optimizer AdamW \
# --log_dir A_resize

# B(wait)
# CUDA_VISIBLE_DEVICES=2 python train_partseg.py \
# --model pointnet2_part_seg_msg_unet  \
# --gpu 2 \
# --optimizer Adam \
# --log_dir B 

# C(wait)
# CUDA_VISIBLE_DEVICES=2 python train_partseg.py \
# --model pointnet2_part_seg_msg_fe  \
# --gpu 2 \
# --optimizer Adam \
# --log_dir C 

# D(wait)
# CUDA_VISIBLE_DEVICES=2 python train_partseg.py \
# --model pointnet2_part_seg_msg_fdr  \
# --gpu 2 \
# --optimizer Adam \
# --log_dir D 

# BD(wait)
# CUDA_VISIBLE_DEVICES=2 python train_partseg.py \
# --model pointnet2_part_seg_msg_unet_fdr  \
# --gpu 2 \
# --optimizer Adam \
# --log_dir BD

# CD(wait)
# CUDA_VISIBLE_DEVICES=2 python train_partseg.py \
# --model pointnet2_part_seg_msg_fe_fdr  \
# --gpu 2 \
# --optimizer Adam \
# --log_dir CD 

# ACD
# CUDA_VISIBLE_DEVICES=5 python train_partseg_strategy.py \
# --model pointnet2_part_seg_msg_fe_fdr  \
# --gpu 5 \
# --optimizer AdamW \
# --log_dir ACD 

# BCD
# CUDA_VISIBLE_DEVICES=7 python train_partseg.py \
# --model pointnet2_part_seg_msg_unet_fe_fdr  \
# --gpu 7 \
# --optimizer Adam \
# --log_dir BCD 

# A optimizer
# CUDA_VISIBLE_DEVICES=3 python train_partseg_optimizer.py \
# --model pointnet2_part_seg_msg  \
# --gpu 3 \
# --optimizer AdamW \
# --log_dir A_optimizer

# A augument
# CUDA_VISIBLE_DEVICES=6 python train_partseg_augument.py \
# --model pointnet2_part_seg_msg  \
# --gpu 6 \
# --optimizer Adam \
# --log_dir A_augument

# A steplr
# CUDA_VISIBLE_DEVICES=0 python train_partseg_steplr.py \
# --model pointnet2_part_seg_msg  \
# --gpu 0 \
# --optimizer Adam \
# --log_dir A_steplr

# origin 
# CUDA_VISIBLE_DEVICES=0 python train_partseg.py \
# --model pointnet2_part_seg_msg  \
# --gpu 0 \
# --optimizer Adam \
# --log_dir origin

# A_optimizer_steplr
# CUDA_VISIBLE_DEVICES=1 python  train_partseg_optimizer_steplr.py \
# --model pointnet2_part_seg_msg  \
# --gpu 1 \
# --optimizer AdamW \
# --log_dir A_optimizer_steplr

# A_optimizer_augment
CUDA_VISIBLE_DEVICES=2 python  train_partseg_optimizer_augment.py \
--model pointnet2_part_seg_msg  \
--gpu 2 \
--optimizer AdamW \
--log_dir A_optimizer_augment

# A_steplr_augment
CUDA_VISIBLE_DEVICES=3 python  train_partseg_steplr_augment.py \
--model pointnet2_part_seg_msg  \
--gpu 3 \
--optimizer Adam \
--log_dir A_steplr_augment