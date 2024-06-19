#!/bin/bash
# kill last traininig processes within current bash
pids=$(ps -f| grep "python" | grep -v grep | awk '{print $2}')
if [ ! -z "$pids" ]; then
    kill -9 $pids;
fi
# make sure we are under current exp directory
cur_path=$(dirname $(readlink -f $0))
cd $cur_path

if [ -z "$1" ]; then
    gpus=0
else
    gpus=$1
fi
if [ -z "$2" ]; then
    exp_name=$(basename $cur_path)_$(date +"%m%d%H%M%S")
else
    exp_name=$2
fi
# train
PYTHONPATH=../../:$PATHPATH ../../curve/tools/dist_train.sh ./config.py $gpus --seed 0 # --deterministic
# Test
PYTHONPATH=../../:$PATHPATH ../../curve/tools/dist_test.sh ./config.py work_dirs/latest.pth $gpus --out work_dirs/$exp_name.pkl --eval F1

