set -x

bs=64
lr=0.0001
drop=0
gcndrop=0

ds=listops

#tree=gold
# tree=ltr
tree=mst-lstm

layers=1
budget=0
dim=50

build/tagger --dataset ${ds} \
             --batch-size ${bs} \
             --drop ${drop} \
             --dim ${dim} \
             --gcn-drop ${gcndrop} \
             --gcn-layers ${layers} \
             --max-iter 100 \
             --tree ${tree} \
             --budget ${budget} \
             --projective \
             --patience 5 \
             --lr ${lr} \
             --save-prefix models/tag_${ds} \
             --mlflow-experiment 17 \
             --mlflow-host 193.136.221.135 \
             --mlflow-name projrt-${tree}-b${budget} \
             --dynet-mem 3000 \
             --sparsemap-eta 1 \
             --sparsemap-max-iter 1 \
             --sparsemap-residual-thr 1e-4;
#             --sparsemap-max-active-set-iter 50 \
#             --sparsemap-eta 1 \
#             --sparsemap-max-iter 1 \
#             --sparsemap-max-iter 10 \
#             --sparsemap-eta 1 \
#             --sparsemap-max-iter-bw 2 \
#             --sparsemap-max-iter 1 \

#done
