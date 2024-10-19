#!/bin/bash
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=2:0:0
#PBS -l select=1:ncpus=1:mem=16gb:scratch_local=32gb:plzen=True
#PBS -N GS2

HOME=/storage/plzen1/home/jivl/ANLP/anlp-2024_kimlova_vladimira
cd $SCRATCHDIR

cp -r $HOME/* $SCRATCHDIR

CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:21.03-py3.SIF
PYTHON_SCRIPT=run_cv02.py

ls /cvmfs/singularity.metacentrum.cz

singularity run $CONTAINER pip install -r /storage/plzen1/home/jivl/ANLP/anlp-2024_kimlova_vladimira/requirements.txt --user
singularity run $CONTAINER python -m wandb login --relogin ff0893cd7836ab91e9386aa470ed0837f2479f9b


echo "Running with random_emb=$RANDOM_EMB, emb_training=$EMB_TRAINING, emb_projection=$EMB_PROJECTION, vocab_size=$VOCAB_SIZE, final_metric=$FINAL_METRIC, lr=$LR, optimizer=$OPTIMIZER, batch_size=$BATCH_SIZE, lr_scheduler=$LR_SCHEDULER"
singularity run $CONTAINER python $PYTHON_SCRIPT --random_emb "$RANDOM_EMB" --emb_training "$EMB_TRAINING" --emb_projection "$EMB_PROJECTION" --vocab_size "$VOCAB_SIZE" --final_metric "$FINAL_METRIC" --lr "$LR" --optimizer "$OPTIMIZER" --batch_size "$BATCH_SIZE" --lr_scheduler "$LR_SCHEDULER"

#PBS -l select=1:ncpus=1:ngpus=1:mem=40000mb:scratch_local=40000mb:cl_adan=True
#singularity run --nv $CONTAINER python $PYTHON_SCRIPT $ARG_1 $ARG_2
