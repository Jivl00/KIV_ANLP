#!/bin/bash

# Best mean model
# {"model":"mean","batches":2000,"batch_size":128,"lr":0.001,"activation":"gelu","gradient_clip":0.5,"proj_size":100,"seq_len":100,"vocab_size":20000,"emb_training":1,"random_emb":0,"emb_projection":1,"device":"cpu","cnn_architecture":"0","n_kernel":0,"learning_rate":0.001,"num_of_params":6031003}


# Best cnn model
# cnn_architecture: A, displayName: {"model":"cnn","batches":2000,"batch_size":128,"lr":0.0001,"activation":"relu","gradient_clip":0.5,"proj_size":100,"seq_len":100,"vocab_size":20000,"emb_training":1,"random_emb":0,"emb_projection":1,"device":"cpu","cnn_architecture":"A","n_kernel":64,"learning_rate":0.0001,"hidden_size":500,"num_of_params":15633471}
# cnn_architecture: B, displayName: {"model":"cnn","batches":1000,"batch_size":128,"lr":0.001,"activation":"relu","gradient_clip":0.5,"proj_size":100,"seq_len":100,"vocab_size":20000,"emb_training":1,"random_emb":0,"emb_projection":1,"device":"cpu","cnn_architecture":"B","n_kernel":64,"learning_rate":0.001,"hidden_size":970,"num_of_params":15561815}
# cnn_architecture: C, displayName: {"model":"cnn","batches":10000,"batch_size":128,"lr":0.0001,"activation":"relu","gradient_clip":0.5,"proj_size":100,"seq_len":100,"vocab_size":20000,"emb_training":1,"random_emb":0,"emb_projection":1,"device":"cpu","cnn_architecture":"C","n_kernel":64,"learning_rate":0.0001,"hidden_size":35020,"num_of_params":12952415}


# Run the best models 10 times
# Best mean model
for i in {1..20}; do
    qsub -v MODEL="mean",BATCHES="2000",BATCH_SIZE="128",LR="0.001",ACTIVATION="gelu",RANDOM_EMB="0",EMB_TRAINING="1",EMB_PROJECTION="1",VOCAB_SIZE="20000",PROJ_SIZE="100",SEQ_LEN="100",N_KERNEL="0",CNN_CONFIG="0" run_GS_on_meta.sh
done

# Best cnn model
for i in {1..30}; do
    qsub -v MODEL="cnn",CNN_CONFIG="A",N_KERNEL="64",BATCHES="2000",BATCH_SIZE="128",ACTIVATION="relu",LR="0.0001",RANDOM_EMB="0",EMB_TRAINING="1",EMB_PROJECTION="1",VOCAB_SIZE="20000",PROJ_SIZE="100",SEQ_LEN="100" run_GS_on_meta.sh
    qsub -v MODEL="cnn",CNN_CONFIG="B",N_KERNEL="64",BATCHES="1000",BATCH_SIZE="128",ACTIVATION="relu",LR="0.001",RANDOM_EMB="0",EMB_TRAINING="1",EMB_PROJECTION="1",VOCAB_SIZE="20000",PROJ_SIZE="100",SEQ_LEN="100" run_GS_on_meta.sh
    qsub -v MODEL="cnn",CNN_CONFIG="C",N_KERNEL="64",BATCHES="10000",BATCH_SIZE="128",ACTIVATION="relu",LR="0.0001",RANDOM_EMB="0",EMB_TRAINING="1",EMB_PROJECTION="1",VOCAB_SIZE="20000",PROJ_SIZE="100",SEQ_LEN="100" run_GS_on_meta.sh
done


