#!/usr/bin/bash
CUDA_VISIBLE_DEVICES=$2 python run_clm.py --model_type gpt2 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --do_train --do_eval --output_dir lm-wiki40b-$1 --dataset_name wiki40b --dataset_config_name $1 --tokenizer_name tokenizer_wiki40b_$1/ --fp16 --report_to wandb --evaluation_strategy steps --seed 43 --num_train_epochs 25 --save_steps 2000
