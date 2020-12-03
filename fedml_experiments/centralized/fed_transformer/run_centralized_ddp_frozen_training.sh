#!/usr/bin/env bash

NPROC_PER_NODE=$1
NNODE=$2
NODE_RANK=$3
MASTER_ADDR=$4
MASTER_PORT=$5
DATASET=$6
DATA_DIR=$7
LR=$8
fine_tune_layer_num=$9
task_specific_layer_num=${10}

python -m torch.distributed.launch \
--nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODE --node_rank=$NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
main_vit_frozen.py \
--is_distributed 1 \
--lr $LR \
--dataset $DATASET \
--data_dir $DATA_DIR \
--fine_tune_layer_num $fine_tune_layer_num \
--task_specific_layer_num $task_specific_layer_num