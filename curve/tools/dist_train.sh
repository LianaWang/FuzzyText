#!/usr/bin/env bash
export OMP_NUM_THREADS=1

CONFIG=$1
GPUS=${2:-"0"}
export CUDA_VISIBLE_DEVICES="$GPUS"
GPU_NUM=$(((${#GPUS}+1)/2))
echo "using $GPU_NUM gpus with number $GPUS"

python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=$RANDOM \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
