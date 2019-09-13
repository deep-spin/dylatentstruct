set -x

bs=64
lr=0.0005
drop=.33

# ds=en_ewt
#ds=de_gsd
ds=ru_gsd

tree=mst-lstm
#tree=ltr
layers=1
budget=8

#for tree in ltr gold mst-lstm;
#do

#for iter in 1 2 3;
#do

build/tagger --dataset ${ds} \
             --batch-size ${bs} \
             --drop ${drop} \
             --gcn-layers ${layers} \
             --max-iter 50 \
             --tree ${tree} \
             --budget ${budget} \
             --patience 5 \
             --lr ${lr} \
             --save-prefix models/tag_${ds} \
             --mlflow-experiment 13 \
             --mlflow-host 193.136.221.135 \
             --mlflow-name tag-${tree}-${layers} \
             --sparsemap-max-iter 10 \
             --sparsemap-eta 0.5 \
             --dynet-mem 1024;
#             --sparsemap-max-iter 1 \
#             --sparsemap-max-iter-bw 2 \
#             --sparsemap-max-active-set-iter 50 \
#             --sparsemap-residual-thr 1e-4 \

#done
