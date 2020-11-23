# the best lr = 0.001, test accuracy is = 0.9661, training accuracy is 0.8654


# 96.36; 86.27
python ./main_vit.py --lr 0.01 --dataset cifar10 --data_dir ./../../../data/cifar10

# 0.9576
python ./main_vit.py --lr 0.03 --dataset cifar10 --data_dir ./../../../data/cifar10


# 0.9654
python ./main_vit.py --lr 0.001 --dataset cifar10 --data_dir ./../../../data/cifar10

# 0.9657
python ./main_vit.py --lr 0.003 --dataset cifar10 --data_dir ./../../../data/cifar10
