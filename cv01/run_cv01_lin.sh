#!/bin/bash

models="dense cnn"
lrs="0.1 0.01 0.001 0.0001 0.00001"
opts="sgd adam"
dps="0 0.1 0.3 0.5"




for m in $models; do
    for l in $lrs; do
        for o in $opts; do
            for d in $dps; do
                echo "model=$m, lr=$l, optimizer=$o, dp=$d"
                qsub -v MODEL="$m",LR="$l",OPTIMIZER="$o",DP="$d" run_GS_on_meta.sh
            done
        done
    done
done