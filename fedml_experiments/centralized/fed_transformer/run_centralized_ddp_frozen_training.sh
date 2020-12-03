#!/usr/bin/env bash

NPROC_PER_NODE=$1
NNODE=$2
NODE_RANK=$3
MASTER_ADDR=$4
MASTER_PORT=$5
DATASET=$6
DATA_DIR=$7
if_name=8
LR=$9
fine_tune_layer_num=${10}
task_specific_layer_num=${11}


python -m torch.distributed.launch \
--nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODE --node_rank=$NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
main_vit_frozen.py \
--is_distributed 1 \
--if_name $if_name \
--lr $LR \
--dataset $DATASET \
--data_dir $DATA_DIR \
--fine_tune_layer_num $fine_tune_layer_num \
--task_specific_layer_num $task_specific_layer_num