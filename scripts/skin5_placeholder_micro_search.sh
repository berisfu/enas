#!/bin/bash

export PYTHONPATH="$(pwd)"

python2 src/skin5_placeholder/main.py \
  --data_format="NCHW" \
  --search_for="micro" \
  --reset_output_dir \
  --data_path="data/skin5/" \
  --output_dir="outputs_skin5_placeholder_micro_search" \
  --batch_size=4 \
  --num_epochs=300 \
  --log_every=100 \
  --eval_every_epochs=1 \
  --child_use_aux_heads=False \
  --child_num_layers=6 \
  --child_out_filters=20 \
  --child_l2_reg=1e-4 \
  --child_lr_dec_every=20 \
  --child_num_branches=5 \
  --child_num_cells=5 \
  --child_keep_prob=0.90 \
  --child_drop_path_keep_prob=0.60 \
  --child_lr_cosine=False \
  --child_lr_max=0.05 \
  --child_lr_min=0.00005 \
  --child_lr_dec_rate=0.9991 \
  --child_lr_T_0=10 \
  --child_lr_T_mul=2 \
  --controller_training \
  --controller_search_whole_channels \
  --controller_entropy_weight=0.0001 \
  --controller_train_every=1 \
  --controller_sync_replicas \
  --controller_num_aggregate=10 \
  --controller_train_steps=30 \
  --controller_lr=0.0035 \
  --controller_tanh_constant=1.50 \
  --controller_op_tanh_reduce=2.5 \
  "$@"
