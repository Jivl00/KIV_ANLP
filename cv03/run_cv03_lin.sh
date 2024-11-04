#!/bin/bash


model="cnn mean"
cnn_config="A B C"
n_kernel="64"

batches="2000 1000"
batch_size="32 128"
activation="relu gelu"
lrs="0.001 0.0001 0.00001 0.000001"

random_emb="0 1"
emb_training="0 1"
emb_projection="0 1"
vocab_size="20000"
proj_size="100"
seq_len="100"


# grad_clip="0.5"

# 2 for loops depending on the model
# CNN
model="cnn"
for conf in $cnn_config; do
    for n_k in $n_kernel; do
        for b in $batches; do
            for bs in $batch_size; do
                for a in $activation; do
                    for l in $lrs; do
                        for re in $random_emb; do
                            for et in $emb_training; do
                                for ep in $emb_projection; do
                                    for vs in $vocab_size; do
                                        for ps in $proj_size; do
                                            for sl in $seq_len; do
                                                echo "model=$model, cnn_config=$conf, n_kernel=$n_k, batches=$b, batch_size=$bs, activation=$a, lr=$l, random_emb=$re, emb_training=$et, emb_projection=$ep, vocab_size=$vs, proj_size=$ps, seq_len=$sl"
                                                qsub -v MODEL="$model",CNN_CONFIG="$conf",N_KERNEL="$n_k",BATCHES="$b",BATCH_SIZE="$bs",ACTIVATION="$a",LR="$l",RANDOM_EMB="$re",EMB_TRAINING="$et",EMB_PROJECTION="$ep",VOCAB_SIZE="$vs",PROJ_SIZE="$ps",SEQ_LEN="$sl" run_GS_on_meta.sh
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

# MEAN
model="mean"
for b in $batches; do
    for bs in $batch_size; do
        for a in $activation; do
            for l in $lrs; do
                for re in $random_emb; do
                    for et in $emb_training; do
                        for ep in $emb_projection; do
                            for vs in $vocab_size; do
                                for ps in $proj_size; do
                                    for sl in $seq_len; do
                                        echo "model=$model, batches=$b, batch_size=$bs, activation=$a, lr=$l, random_emb=$re, emb_training=$et, emb_projection=$ep, vocab_size=$vs, proj_size=$ps, seq_len=$sl"
                                        qsub -v MODEL="$model",BATCHES="$b",BATCH_SIZE="$bs",ACTIVATION="$a",LR="$l",RANDOM_EMB="$re",EMB_TRAINING="$et",EMB_PROJECTION="$ep",VOCAB_SIZE="$vs",PROJ_SIZE="$ps",SEQ_LEN="$sl",N_KERNEL="0",CNN_CONFIG="0" run_GS_on_meta.sh
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

# qsub -v MODEL="cnn",CNN_CONFIG="A",N_KERNEL="64",BATCHES="100000",BATCH_SIZE="32",ACTIVATION="relu",LR="0.001",RANDOM_EMB="0",EMB_TRAINING="0",EMB_PROJECTION="0",VOCAB_SIZE="20000",PROJ_SIZE="100",SEQ_LEN="100" run_GS_on_meta.sh