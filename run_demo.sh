#!/usr/bin/env bash

DATASET=flower #cub cifar100 imagenet_sub
GPU=0
NET=resnet18 # resnet32
LR=1e-4 
EPOCH=51 
SAVE=50
LOSS=triplet 
TASK=6
BASE=17
SEED=1

for Method in Finetuning #LwF EWC MAS
do
for Tradeoff in 0 # 1 1e7 1e6
do

NAME=${Method}_${Tradeoff}_${DATASET}_${LOSS}_${NET}_${LR}_${EPOCH}epochs_task${TASK}_base${BASE}_seed${SEED}_pre

python train.py -base ${BASE} -seed ${SEED} -task ${TASK} -data ${DATASET} -tradeoff ${Tradeoff} -exp ${Tradeoff} -net ${NET} -method ${Method} \
-lr ${LR} -dim 512  -num_instances 8 -BatchSize 32 -loss ${LOSS}  -epochs ${EPOCH} -log_dir ${DATASET}_seed${SEED}/${NAME}  \
-save_step ${SAVE} -gpu ${GPU} 


python test.py  -seed ${SEED} -base ${BASE} -task ${TASK} -epochs ${EPOCH} -data ${DATASET} -gpu ${GPU} -method ${Method} -r  \
checkpoints/${DATASET}_seed${SEED}/${NAME} >./results/${DATASET}/${NAME}_old_mean.txt 


for SIGMA_TEST in 0.30 
do

python test.py -seed ${SEED} -base ${BASE} -task ${TASK} -epochs ${EPOCH} -data ${DATASET} -gpu ${GPU} -method ${Method} -r  \
checkpoints/${DATASET}_seed${SEED}/${NAME} -mapping_test -sigma_test ${SIGMA_TEST} \
>./results/${DATASET}/${NAME}_SDC_sigma_test${SIGMA_TEST}.txt 

done

done
done

