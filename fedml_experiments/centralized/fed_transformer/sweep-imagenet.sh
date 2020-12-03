
# start training at 18:00pm
gpu=$1
fine_tune_layer_num=$2
task_specific_layer_num=$3
lr=$4

# lr = 0.03, 0.01, 0.003, 0.06
# sh sweep-imagenet.sh 0 0 0 0.03
# sh sweep-imagenet.sh 1 0 0 0.01
# sh sweep-imagenet.sh 2 0 0 0.003
# sh sweep-imagenet.sh 3 0 0 0.06
python ./main_vit_frozen.py --lr $lr --dataset imagenet --data_dir /home/chaoyanghe/sourcecode/dataset/cv/ImageNet --local_rank $gpu --fine_tune_layer_num $fine_tune_layer_num --task_specific_layer_num $task_specific_layer_num
