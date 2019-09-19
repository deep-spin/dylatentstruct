set -x

bs=64
lr=0.0001
drop=0
gcndrop=0

ds=listops

#tree=gold
tree=ltr
# tree=mst-lstm

layers=1
budget=5

build/tagger --dataset ${ds} \
             --batch-size ${bs} \
             --drop ${drop} \
             --gcn-drop ${gcndrop} \
             --gcn-layers ${layers} \
             --max-iter 100 \
             --tree ${tree} \
             --budget ${budget} \
             --patience 5 \
             --lr ${lr} \
             --save-prefix models/tag_${ds} \
             --mlflow-experiment 17 \
             --mlflow-host 193.136.221.135 \
             --mlflow-name tag-${tree}-${layers} \
             --dynet-mem 3000 \
             --sparsemap-eta 0.5 \
             --sparsemap-max-iter 10 \
             --sparsemap-residual-thr 1e-4;
#             --sparsemap-max-iter 10 \
#             --sparsemap-max-active-set-iter 50 \
#             --sparsemap-eta 1 \
#             --sparsemap-max-iter-bw 2 \
#             --sparsemap-max-iter 1 \

#done
