## Frozen Backbone (task specific layer = Transformer Encoder)
nohup sh sweep-cifar10.sh > ./sweep.txt 2>&1 &
nohup sh sweep-cifar100.sh > ./sweep.txt 2>&1 &

## Fine Tune
Single Process
```
python ./main_vit_fine_tune.py --lr 0.003 --dataset cifar10 --data_dir ./../../../data/cifar10
```

DDP Training
```
# sh run_centralized_ddp_train.sh 4 1 0 127.0.0.1 11111

nohup sh run_centralized_ddp_train.sh 8 1 0 127.0.0.1 11111 cifar10 ./../../../data/cifar10 0.001 > ./machine1.txt 2>&1 &
nohup sh run_centralized_ddp_train.sh 8 1 0 127.0.0.1 11111 cifar10 ./../../../data/cifar10 0.003 > ./machine1.txt 2>&1 &
```

```
# kill all processes
kill $(ps aux | grep "ddp_demo.py" | grep -v grep | awk '{print $2}')
```