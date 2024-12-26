accelerate launch train_test_ds.py \
    --output_dir output2/test \
    --do_train \
    --dataloader_num_workers 8 \
    --remove_unused_columns False \
    --per_device_train_batch_size 6 \
    --max_steps 10000 \
    --learning_rate 0.00005 \
    --report_to "wandb" \
    --logging_steps 1 \
    --save_steps 100 \
    --lr_scheduler_type "linear" \
    --warmup_steps 200 \
    --fp16 \
    --gradient_accumulation_steps 8 \
    --train_datasets "avs_object" \
    --rephrase_weight 0.1 \
    --lora_name "no" \
    --clip_resize_wo_crop True \
    --roi True \
    --no_mask True \
    --add_audio_encoder True