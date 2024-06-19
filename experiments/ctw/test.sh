#!/bin/bash
# usage1: ./test.sh 0,1,2,3 
# usage2: ./test.sh 0,1,2,3 custom_pkl_filename
pids=$(ps -f| grep "python" | grep -v grep | awk '{print $2}')
if [ ! -z "$pids" ]; then
    kill -9 $pids;
fi

cur_path=$(dirname $(readlink -f $0))
cd $cur_path

if [ -z "$1" ]; then
    gpus=0
else
    gpus=$1
fi

exp_name=$(basename $cur_path)_$(date +"%m%d%H%M")
echo "test on gpus=${gpus} and output '${exp_name}.pkl/json' files"
PYTHONPATH=../../:$PATHPATH ../../curve/tools/dist_test.sh ./config.py latest.pth $gpus --out $exp_name.pkl --eval F1
