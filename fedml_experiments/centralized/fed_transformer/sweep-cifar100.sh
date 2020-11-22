# the best lr = 0.001, test accuracy is = 0.9661, training accuracy is 0.8654

python ./main_vit.py --lr 0.01 --dataset cifar100 --data_dir ./../../../data/cifar100
python ./main_vit.py --lr 0.001 --dataset cifar100 --data_dir ./../../../data/cifar100
python ./main_vit.py --lr 0.0001 --dataset cifar100 --data_dir ./../../../data/cifar100
python ./main_vit.py --lr 0.1 --dataset cifar100 --data_dir ./../../../data/cifar100
python ./main_vit.py --lr 0.00001 --dataset cifar100 --data_dir ./../../../data/cifar100