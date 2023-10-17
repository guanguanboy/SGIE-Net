#!/usr/bin/env bash

CONFIG=$1

python -m torch.distributed.launch --nproc_per_node=3 --master_port=4326 basicsr/train_sam_finetune.py -opt $CONFIG --launcher pytorch