#!/usr/bin/env bash

work_path=$(dirname $(readlink -f $0))
cd ${work_path}
python main.py --dataset=$1 --optimizer=$2 --lr=$3 --epoch=$4 --wd=$5
