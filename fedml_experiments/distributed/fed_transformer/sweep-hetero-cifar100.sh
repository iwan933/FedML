# sweep-hetero
# nohup sh sweep-hetero-cifar100.sh > sweep-hetero-cifar100.txt 2>&1 &
sh run_fed_transformer.sh 10 10 1 8 transformer hetero 100 1 64 0.0001 cifar100 "./../../../data/cifar100" 0 0

sh run_fed_transformer.sh 10 10 1 8 transformer hetero 100 1 64 0.0003 cifar100 "./../../../data/cifar100" 0 0

sh run_fed_transformer.sh 10 10 1 8 transformer hetero 100 1 64 0.001 cifar100 "./../../../data/cifar100" 0 0

sh run_fed_transformer.sh 10 10 1 8 transformer hetero 100 1 64 0.003 cifar100 "./../../../data/cifar100" 0 0

sh run_fed_transformer.sh 10 10 1 8 transformer hetero 100 1 64 0.1 cifar100 "./../../../data/cifar100" 0 0

sh run_fed_transformer.sh 10 10 1 8 transformer hetero 100 1 64 0.3 cifar100 "./../../../data/cifar100" 0 0

sh run_fed_transformer.sh 10 10 1 8 transformer hetero 100 1 64 0.03 cifar100 "./../../../data/cifar100" 0 0

sh run_fed_transformer.sh 10 10 1 8 transformer hetero 100 1 64 0.01 cifar100 "./../../../data/cifar100" 0 0

