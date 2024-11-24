#!/bin/bash
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=1:ngpus=1:mem=32gb:scratch_local=32gb
#PBS -N GS4CZSGPU

HOME=/storage/plzen1/home/jivl/ANLP/anlp-2024_kimlova_vladimira
cd $SCRATCHDIR

cp -r $HOME/* $SCRATCHDIR

CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:21.03-py3.SIF
PYTHON_SCRIPT=cv04/main04.py

ls /cvmfs/singularity.metacentrum.cz

singularity run $CONTAINER pip install -r /storage/plzen1/home/jivl/ANLP/anlp-2024_kimlova_vladimira/requirements.txt --user
singularity run $CONTAINER python -m wandb login --relogin ff0893cd7836ab91e9386aa470ed0837f2479f9b

# qsub -v L2_ALPHA="$l",LR="$lr" run_GS_on_meta.sh

# singularity run $CONTAINER python $PYTHON_SCRIPT \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --model_type CZERT \
#     --data_dir cv04/data \
#     --labels cv04/data/labels.txt \
#     --output_dir output \
#     --do_predict \
#     --do_train \
#     --do_eval \
#     --eval_steps 100 \
#     --logging_steps 50 \
#     --learning_rate 0.0001 \
#     --warmup_steps 4000 \
#     --num_train_epochs 5 \
#     --dropout_probs 0.05 \
#     --l2_alpha 0.01 \
#     --task NER


# singularity run $CONTAINER python $PYTHON_SCRIPT \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --model_type SLAVIC \
#     --data_dir cv04/data \
#     --labels cv04/data/labels.txt \
#     --output_dir output \
#     --do_predict \
#     --do_train \
#     --do_eval \
#     --eval_steps 100 \
#     --logging_steps 50 \
#     --learning_rate 0.0001 \
#     --warmup_steps 4000 \
#     --num_train_epochs 5 \
#     --dropout_probs 0.05 \
#     --l2_alpha 0.01 \
#     --task NER


singularity run --nv $CONTAINER python3 $PYTHON_SCRIPT \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --model_type CZERT \
    --data_dir cv04/data-mt \
    --labels cv04/data-mt/labels.txt \
    --output_dir output \
    --do_predict \
    --do_train \
    --do_eval \
    --eval_steps 300 \
    --logging_steps 50 \
    --learning_rate 0.0001 \
    --warmup_steps 4000 \
    --num_train_epochs 10 \
    --dropout_probs 0.05 \
    --l2_alpha 0.01 \
    --task TAGGING


# singularity run --nv $CONTAINER python3 $PYTHON_SCRIPT \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --model_type SLAVIC \
#     --data_dir cv04/data-mt \
#     --labels cv04/data-mt/labels.txt \
#     --output_dir output \
#     --do_predict \
#     --do_train \
#     --do_eval \
#     --eval_steps 300 \
#     --logging_steps 50 \
#     --learning_rate 0.0001 \
#     --warmup_steps 4000 \
#     --num_train_epochs 10 \
#     --dropout_probs 0.05 \
#     --l2_alpha 0.01 \
#     --task TAGGING


clean_scratch