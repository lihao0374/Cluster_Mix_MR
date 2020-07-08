#!/bin/bash
set -x
set -e


cd /home/ma-user/work/Cluster_Mix_No_RA/
python -W ignore main_moco.py \
--moxing=false \
--train_url=../mul_pos \
--data_dir=/home/ma-user/work/data/sub-imagenet \
--moco_dim=128 \
--moco_k=66560 \
--moco_m=0.999 \
--moco_t=0.2 \
--mlp=true \
--aug_plus=true \
--decay_method=cos \
--init_lr=0.03 \
--batch_size=256 \
--num_workers=32 \
--end_epoch=200 \
--dist=true \
--nodes_num=1 \
--node_rank=0 \
--report_freq=10 \
--subgroup=2 \
--cluster_center=100 \
--unpdate_label=10 \
--alpha=1 \
--prob=0.8 \
--mix=true \
--use_RA = False
# --resume=true \
