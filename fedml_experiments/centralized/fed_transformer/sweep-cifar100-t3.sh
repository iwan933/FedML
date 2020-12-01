# the best lr = 0.001, test accuracy is = 0.9661, training accuracy is 0.8654
python ./main_vit_frozen.py --lr 0.03 --dataset cifar100 --data_dir ./../../../data/cifar100 --local_rank 2 --task_specific_layer_type 3

python ./main_vit_frozen.py --lr 0.01 --dataset cifar100 --data_dir ./../../../data/cifar100 --local_rank 2 --task_specific_layer_type 3

# 0.8431, 0.8645
python ./main_vit_frozen.py --lr 0.003 --dataset cifar100 --data_dir ./../../../data/cifar100 --local_rank 2 --task_specific_layer_type 3
python ./main_vit_frozen.py --lr 0.001 --dataset cifar100 --data_dir ./../../../data/cifar100 --local_rank 2 --task_specific_layer_type 3

python ./main_vit_frozen.py --lr 0.0003 --dataset cifar100 --data_dir ./../../../data/cifar100 --local_rank 2 --task_specific_layer_type 3

python ./main_vit_frozen.py --lr 0.0001 --dataset cifar100 --data_dir ./../../../data/cifar100 --local_rank 2 --task_specific_layer_type 3