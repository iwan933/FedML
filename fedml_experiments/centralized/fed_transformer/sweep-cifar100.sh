
# start training at 18:00pm
gpu=$1
fine_tune_layer_num=$2
task_specific_layer_num=$3

python ./main_vit_frozen.py --lr 0.03 --dataset cifar100 --data_dir ./../../../data/cifar100 --local_rank $gpu --fine_tune_layer_num $fine_tune_layer_num --task_specific_layer_num $task_specific_layer_num

# 0.8895, 0.9261
python ./main_vit_frozen.py --lr 0.01 --dataset cifar100 --data_dir ./../../../data/cifar100 --local_rank $gpu --fine_tune_layer_num $fine_tune_layer_num --task_specific_layer_num $task_specific_layer_num

# 0.8431, 0.8645
python ./main_vit_frozen.py --lr 0.003 --dataset cifar100 --data_dir ./../../../data/cifar100 --local_rank $gpu --fine_tune_layer_num $fine_tune_layer_num --task_specific_layer_num $task_specific_layer_num
python ./main_vit_frozen.py --lr 0.001 --dataset cifar100 --data_dir ./../../../data/cifar100 --local_rank $gpu --fine_tune_layer_num $fine_tune_layer_num --task_specific_layer_num $task_specific_layer_num

python ./main_vit_frozen.py --lr 0.0003 --dataset cifar100 --data_dir ./../../../data/cifar100 --local_rank $gpu --fine_tune_layer_num $fine_tune_layer_num --task_specific_layer_num $task_specific_layer_num

python ./main_vit_frozen.py --lr 0.0001 --dataset cifar100 --data_dir ./../../../data/cifar100 --local_rank $gpu --fine_tune_layer_num $fine_tune_layer_num --task_specific_layer_num $task_specific_layer_num