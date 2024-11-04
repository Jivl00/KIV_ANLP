#!/bin/bash
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=4:0:0
#PBS -l select=1:ncpus=1:mem=32gb:scratch_local=32gb
#PBS -N GS3

HOME=/storage/plzen1/home/jivl/ANLP/anlp-2024_kimlova_vladimira
cd $SCRATCHDIR

cp -r $HOME/* $SCRATCHDIR

CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:21.03-py3.SIF
PYTHON_SCRIPT=run_cv03.py

ls /cvmfs/singularity.metacentrum.cz

singularity run $CONTAINER pip install -r /storage/plzen1/home/jivl/ANLP/anlp-2024_kimlova_vladimira/requirements.txt --user
singularity run $CONTAINER python -m wandb login --relogin ff0893cd7836ab91e9386aa470ed0837f2479f9b


echo "Running with model=$MODEL, batches=$BATCHES, batch_size=$BATCH_SIZE, lr=$LR, activation=$ACTIVATION", random_emb=$RANDOM_EMB, emb_training=$EMB_TRAINING, emb_projection=$EMB_PROJECTION, vocab_size=$VOCAB_SIZE, proj_size=$PROJ_SIZE, seq_len=$SEQ_LEN, cnn_architecture=$CNN_CONFIG, n_kernel=$N_KERNEL
singularity run $CONTAINER python $PYTHON_SCRIPT --model "$MODEL" --batches "$BATCHES" --batch_size "$BATCH_SIZE" --lr "$LR" --activation "$ACTIVATION" --random_emb "$RANDOM_EMB" --emb_training "$EMB_TRAINING" --emb_projection "$EMB_PROJECTION" --vocab_size "$VOCAB_SIZE" --proj_size "$PROJ_SIZE" --seq_len "$SEQ_LEN" --cnn_architecture "$CNN_CONFIG" --n_kernel "$N_KERNEL"
