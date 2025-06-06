#!/bin/bash


random_emb="0 1"
emb_training="0 1"
emb_projection="0 1"
vocab_size="20000 40000"
final_metric="cos neural"
lrs="0.1 0.01 0.001 0.0001 0.00001"
opts="sgd adam"
batch_size="100 500 1000"
lr_scheduler="step exp"

for re in $random_emb; do
    for et in $emb_training; do
        for ep in $emb_projection; do
            for vs in $vocab_size; do
                for fm in $final_metric; do
                    for l in $lrs; do
                        for o in $opts; do
                            for bs in $batch_size; do
                                for ls in $lr_scheduler; do
                                    echo "random_emb=$re, emb_training=$et, emb_projection=$ep, vocab_size=$vs, final_metric=$fm, lr=$l, optimizer=$o, batch_size=$bs, lr_scheduler=$ls"
                                    qsub -v RANDOM_EMB="$re",EMB_TRAINING="$et",EMB_PROJECTION="$ep",VOCAB_SIZE="$vs",FINAL_METRIC="$fm",LR="$l",OPTIMIZER="$o",BATCH_SIZE="$bs",LR_SCHEDULER="$ls" run_GS_on_meta.sh
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
