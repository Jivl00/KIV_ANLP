#!/bin/bash
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=48:0:0
#PBS -l select=1:ncpus=1:mem=32gb:scratch_local=32gb
#PBS -N GS4

HOME=/storage/plzen1/home/jivl/ANLP/anlp-2024_kimlova_vladimira
cd $SCRATCHDIR

cp -r $HOME/* $SCRATCHDIR

CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:21.03-py3.SIF
PYTHON_SCRIPT=cv04/main04.py

ls /cvmfs/singularity.metacentrum.cz

singularity run $CONTAINER pip install -r /storage/plzen1/home/jivl/ANLP/anlp-2024_kimlova_vladimira/requirements.txt --user
singularity run $CONTAINER python -m wandb login --relogin ff0893cd7836ab91e9386aa470ed0837f2479f9b

# L2_ALPHA=0.01
# LR=0.0001
echo "Running with l2_alpha=$L2_ALPHA, lr=$LR"
# singularity run $CONTAINER python $PYTHON_SCRIPT \
#    --per_device_train_batch_size 8 \
#    --per_device_eval_batch_size 8 \
#    --model_type RNN \
#    --data_dir cv04/data \
#    --labels cv04/data/labels.txt \
#    --output_dir output \
#    --do_predict \
#    --do_train \
#    --do_eval \
#    --eval_steps 200 \
#    --logging_steps 50 \
#    --learning_rate 0.0001 \
#    --warmup_steps 4000 \
#    --num_train_epochs 100 \
#    --no_bias \
#    --dropout_probs 0.05 \
#    --lstm_hidden_dimension 64 \
#    --num_lstm_layers 2 \
#    --embedding_dimension 128 \
#    --task NER \
#    --l2_alpha $L2_ALPHA \
#    --learning_rate $LR

# singularity run $CONTAINER python $PYTHON_SCRIPT \
#    --per_device_train_batch_size 8 \
#    --per_device_eval_batch_size 8 \
#    --model_type RNN \
#    --data_dir cv04/data \
#    --labels cv04/data/labels.txt \
#    --output_dir output \
#    --do_predict \
#    --do_train \
#    --do_eval \
#    --eval_steps 200 \
#    --logging_steps 50 \
#    --learning_rate 0.0001 \
#    --warmup_steps 4000 \
#    --num_train_epochs 100 \
#    --dropout_probs 0.05 \
#    --lstm_hidden_dimension 64 \
#    --num_lstm_layers 2 \
#    --embedding_dimension 128 \
#    --task NER \
#    --l2_alpha $L2_ALPHA \
#    --learning_rate $LR

# singularity run $CONTAINER python $PYTHON_SCRIPT \
#    --model_type LSTM \
#    --data_dir cv04/data \
#    --labels cv04/data/labels.txt \
#    --output_dir output \
#    --do_predict \
#    --do_train \
#    --do_eval \
#    --eval_steps 200 \
#    --eval_dataset_batches 200 \
#    --logging_steps 50 \
#    --learning_rate 0.0001 \
#    --warmup_steps 4000 \
#    --num_train_epochs 100 \
#    --dropout_probs 0.05 \
#    --lstm_hidden_dimension 128 \
#    --num_lstm_layers 2 \
#    --embedding_dimension 128 \
#    --task NER \
#    --no_bias \
#    --l2_alpha $L2_ALPHA \
#    --learning_rate $LR

# singularity run $CONTAINER python $PYTHON_SCRIPT \
#    --model_type LSTM \
#    --data_dir cv04/data \
#    --labels cv04/data/labels.txt \
#    --output_dir output \
#    --do_predict \
#    --do_train \
#    --do_eval \
#    --eval_steps 200 \
#    --eval_dataset_batches 200 \
#    --logging_steps 50 \
#    --learning_rate 0.0001 \
#    --warmup_steps 4000 \
#    --num_train_epochs 100 \
#    --dropout_probs 0.05 \
#    --lstm_hidden_dimension 128 \
#    --num_lstm_layers 2 \
#    --embedding_dimension 128 \
#    --task NER \
#    --l2_alpha $L2_ALPHA \
#    --learning_rate $LR


# singularity run $CONTAINER python $PYTHON_SCRIPT \
#    --model_type RNN \
#    --data_dir cv04/data-mt \
#    --labels cv04/data-mt/labels.txt \
#    --output_dir output \
#    --do_predict \
#    --do_train \
#    --do_eval \
#    --eval_steps 300 \
#    --eval_dataset_batches 200 \
#    --logging_steps 50 \
#    --learning_rate 0.0001 \
#    --warmup_steps 4000 \
#    --num_train_epochs 10 \
#    --no_bias \
#    --dropout_probs 0.05 \
#    --lstm_hidden_dimension 128 \
#    --num_lstm_layers 2 \
#    --embedding_dimension 128 \
#    --task TAGGING \
#    --l2_alpha $L2_ALPHA \
#    --learning_rate $LR

# singularity run $CONTAINER python $PYTHON_SCRIPT \
#    --model_type RNN \
#    --data_dir cv04/data-mt \
#    --labels cv04/data-mt/labels.txt \
#    --output_dir output \
#    --do_predict \
#    --do_train \
#    --do_eval \
#    --eval_steps 300 \
#    --eval_dataset_batches 200 \
#    --logging_steps 50 \
#    --learning_rate 0.0001 \
#    --warmup_steps 4000 \
#    --num_train_epochs 10 \
#    --dropout_probs 0.05 \
#    --lstm_hidden_dimension 128 \
#    --num_lstm_layers 2 \
#    --embedding_dimension 128 \
#    --task TAGGING \
#    --l2_alpha $L2_ALPHA \
#    --learning_rate $LR

# singularity run $CONTAINER python $PYTHON_SCRIPT \
#    --per_device_train_batch_size 8 \
#    --per_device_eval_batch_size 8 \
#    --model_type LSTM \
#    --data_dir cv04/data-mt \
#    --labels cv04/data-mt/labels.txt \
#    --output_dir output \
#    --do_predict \
#    --do_train \
#    --do_eval \
#    --eval_steps 300 \
#    --logging_steps 50 \
#    --learning_rate 0.0001 \
#    --warmup_steps 4000 \
#    --num_train_epochs 10 \
#    --dropout_probs 0.05 \
#    --lstm_hidden_dimension 128 \
#    --num_lstm_layers 2 \
#    --embedding_dimension 128 \
#    --task TAGGING \
#    --no_bias \
#    --l2_alpha $L2_ALPHA \
#    --learning_rate $LR

singularity run $CONTAINER python $PYTHON_SCRIPT \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --model_type LSTM \
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
   --lstm_hidden_dimension 128 \
   --num_lstm_layers 2 \
   --embedding_dimension 128 \
   --task TAGGING \
   --l2_alpha $L2_ALPHA \
   --learning_rate $LR


clean_scratch