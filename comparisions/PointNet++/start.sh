# A(run)
CUDA_VISIBLE_DEVICES=2 python train_partseg_strategy.py \
--model pointnet2_part_seg_msg  \
--gpu 2 \
--optimizer AdamW \
--log_dir A 

# B(wait)
CUDA_VISIBLE_DEVICES=3 python train_partseg.py \
--model pointnet2_part_seg_msg_unet  \
--gpu 3 \
--optimizer Adam \
--log_dir B 

# C(wait)
CUDA_VISIBLE_DEVICES=3 python train_partseg.py \
--model pointnet2_part_seg_msg_fe  \
--gpu 3 \
--optimizer Adam \
--log_dir C 

# D(wait)
CUDA_VISIBLE_DEVICES=3 python train_partseg.py \
--model pointnet2_part_seg_msg_fdr  \
--gpu 3 \
--optimizer Adam \
--log_dir D 

# AB(wait)
CUDA_VISIBLE_DEVICES=3 python train_partseg_strategy.py \
--model pointnet2_part_seg_msg_unet  \
--gpu 3 \
--optimizer AdamW \
--log_dir AB 

# AC(wait)
CUDA_VISIBLE_DEVICES=3 python train_partseg_strategy.py \
--model pointnet2_part_seg_msg_fe  \
--gpu 3 \
--optimizer AdamW \
--log_dir AC 

# AD(wait)
CUDA_VISIBLE_DEVICES=3 python train_partseg_strategy.py \
--model pointnet2_part_seg_msg_fdr  \
--gpu 3 \
--optimizer AdamW \
--log_dir AD

# BC(wait)
CUDA_VISIBLE_DEVICES=3 python train_partseg.py \
--model pointnet2_part_seg_msg_unet_fe  \
--gpu 3 \
--optimizer Adam \
--log_dir BC

# BD(wait)
CUDA_VISIBLE_DEVICES=3 python train_partseg.py \
--model pointnet2_part_seg_msg_unet_fdr  \
--gpu 3 \
--optimizer Adam \
--log_dir BD

# CD(wait)
CUDA_VISIBLE_DEVICES=3 python train_partseg.py \
--model pointnet2_part_seg_msg_fe_fdr  \
--gpu 3 \
--optimizer Adam \
--log_dir CD 

# ABC
CUDA_VISIBLE_DEVICES=3 python train_partseg_strategy.py \
--model pointnet2_part_seg_msg_unet_fe  \
--gpu 3 \
--optimizer AdamW \
--log_dir ABC 

# ABD
CUDA_VISIBLE_DEVICES=3 python train_partseg_strategy.py \
--model pointnet2_part_seg_msg_unet_fdr  \
--gpu 3 \
--optimizer AdamW \
--log_dir ABD

# ACD
CUDA_VISIBLE_DEVICES=3 python train_partseg_strategy.py \
--model pointnet2_part_seg_msg_fe_fdr  \
--gpu 3 \
--optimizer AdamW \
--log_dir ACD 

# BCD
CUDA_VISIBLE_DEVICES=3 python train_partseg.py \
--model pointnet2_part_seg_msg_unet_fe_fdr  \
--gpu 3 \
--optimizer Adam \
--log_dir BCD 

# ABCD
CUDA_VISIBLE_DEVICES=3 python train_partseg_strategy.py \
--model pointnet2_part_seg_msg_unet_fe_fdr  \
--gpu 3 \
--optimizer AdamW \
--log_dir ABCD 
