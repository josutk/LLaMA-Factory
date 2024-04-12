#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../../src/train_bash.py \
    --stage pt \
    --do_train \
    --model_name_or_path josu/gemma-pt-br4 \
    --dataset wiki_pt \
    --dataset_dir /drive/MyDrive/pt_data/data/pt_wiki_corpus.txt \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir ametista \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 1.5 \
    --val_size 0.1 \
    --plot_loss \
    --fp16
