#!/bin/bash

MODEL="RGT"
UPDIM=64

python train.py \
    --seed 1010 \
    --data spider \
    --up_embed_dim 64 \
    --down_embed_dim 300 \
    --up_max_depth 4 \
    --down_max_dist 4 \
    --up_d_model 64 \
    --down_d_model 512 \
    --up_d_ff 256 \
    --down_d_ff 2048 \
    --up_head_num 8 \
    --down_head_num 8 \
    --up_layer_num 6 \
    --down_layer_num 6 \
    --hid_size 300 \
    --dropout 0.1 \
    --max_oov_num 50 \
    --copy 1 \
    --rel_share 1 \
    --k_v_share 0 \
    --mode "concat" \
    --cross_atten "AOD+None" \
    --up_rel "DRD" "DBS" \
    --down_rel "RPR" "LCA" \
    --gpu 0 \
    --lr 1e-4 \
    --epoch 50 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --schedule_step 20 \
    --log Logs/$MODEL_$UPDIM.log \
    --gamma 0.8 \
    --prefix $MODEL_$UPDIM \
    --model $MODEL \
    --min_freq 1 \
    --train_step 10 \
    --eval_step 500 \
    --output Output/$MODEL_$UPDIM.out 
