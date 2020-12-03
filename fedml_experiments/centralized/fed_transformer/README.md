## Frozen Backbone (task specific layer = Transformer Encoder)
nohup sh sweep-cifar10.sh > ./sweep.txt 2>&1 &
nohup sh sweep-cifar100.sh > ./sweep.txt 2>&1 &

## Fine Tune
Single Process
```
python ./main_vit_fine_tune.py --lr 0.003 --dataset cifar10 --data_dir ./../../../data/cifar10
```

```
python ./main_vit_fine_tune.py --lr 0.003 --dataset imagenet --data_dir /home/chaoyanghe/dataset/cv/imagenet

```



DDP Training
```
# sh run_centralized_ddp_train.sh 4 1 0 127.0.0.1 11111 

# sh run_centralized_ddp_frozen_training.sh 4 1 0 127.0.0.1 11111 imagenet /home/chaoyanghe/sourcecode/dataset/cv/ImageNet ib0 0.03 0 0

# sh run_centralized_ddp_frozen_training.sh 4 1 0 127.0.0.1 11111 imagenet /home/chaoyanghe/sourcecode/dataset/cv/ImageNet lo 0.03 0 0

nohup sh run_centralized_ddp_train.sh 8 1 0 192.168.11.1 11111 cifar10 ./../../../data/cifar10 0.001 > ./machine1.txt 2>&1 &
nohup sh run_centralized_ddp_train.sh 4 1 0 192.168.11.2 22222 cifar10 ./../../../data/cifar10 0.003 > ./machine1.txt 2>&1 &
nohup sh run_centralized_ddp_train.sh 8 1 0 192.168.11.1 11111 cifar10 ./../../../data/cifar10 0.03 > ./machine1.txt 2>&1 &
nohup sh run_centralized_ddp_train.sh 8 1 0 192.168.11.1 11111 cifar10 ./../../../data/cifar10 0.01 > ./machine1.txt 2>&1 &

nohup sh run_centralized_ddp_train.sh 4 1 0 192.168.11.2 22222 cifar100 ./../../../data/cifar100 0.03 > ./machine1.txt 2>&1 &
```

```
# kill all processes
kill $(ps aux | grep "main_vit_fine_tune.py" | grep -v grep | awk '{print $2}')
```