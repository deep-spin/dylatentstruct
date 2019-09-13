#!/bin/bash
set -x

bs=64
lr=0.0005
drop=.33

ds=sstb

for buf in ltr,0,0 ltr,1,0 gold,1,0 mst-lstm,1,0 mst-lstm,1,8; do
    IFS=',' read tree layers budget <<< "${buf}"
    smiter=1 && [[ $budget == 0 ]] && smiter=10
    smeta=1 && [[ $budget == 0 ]] && smeta=0.5

    for lr in 0.00025 0.0005 0.001; do

        build/sentclf --dataset ${ds} \
                      --batch-size ${bs} \
                      --drop ${drop} \
                      --gcn-layers ${layers} \
                      --max-iter 50 \
                      --tree ${tree} \
                      --budget ${budget} \
                      --patience 5 \
                      --lr ${lr} \
                      --save-prefix models/tag_${ds} \
                      --mlflow-experiment 14 \
                      --mlflow-host 193.136.221.135 \
                      --mlflow-name sent-${tree}-${layers}-b${budget} \
                      --sparsemap-max-iter ${smiter} \
                      --sparsemap-eta ${smeta} \
                      --dynet-mem 1024 \
                      --sparsemap-max-active-set-iter 50 \
                      --sparsemap-residual-thr 1e-4;
    done
done
