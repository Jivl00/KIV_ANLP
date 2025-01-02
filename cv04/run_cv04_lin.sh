#!/bin/bash


l2_alpha="0.01 0"
learning_rate="0.0001 0.001"
# --no_bias

# for l in $l2_alpha; do
#     for lr in $learning_rate; do
#         echo "l2_alpha=$l, lr=$lr"
#         qsub -v L2_ALPHA="$l",LR="$lr" run_on_meta.sh
#     done
# done

# qsub run_CZS_on_meta.sh

# --freeze_embedding_layer
freeze_first_x_layers="0 2 4 6"
for l in $freeze_first_x_layers; do
    echo "freeze_first_x_layers=$l"
    qsub -v FREEZE_FIRST_X_LAYERS="$l" run_CZERT_on_meta.sh
done


