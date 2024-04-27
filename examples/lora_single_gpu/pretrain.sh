#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../../src/train_bash.py \
    --stage pt \
    --do_train \
    --model_name_or_path josu/gemma-pt-br-galore-layer \
    --dataset c4 \
    --dataset_dir ../../data \
    --finetuning_type full \
    --output_dir gemma_pedregulho1 \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 1000 \
    --eval_steps 50000 \
    --evaluation_strategy steps \
    --learning_rate 3e-5 \
    --num_train_epochs 3.0 \
    --val_size 0.1 \
    --plot_loss \
    --flash_attn True \
    --use_galore True \
    --galore_rank 256 \
    --galore_update_interval 200   \
    --galore_target mlp,self_attn \
    --galore_layerwise \
    --pure_bf16
