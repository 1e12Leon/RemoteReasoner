BATCH_PER_DEVICE=8
GRAD_ACCUM_STEPS=8
MODEL_NAME="./Qwen2.5-VL/Qwen2.5-VL-7B-Instruct/"
DATA_PATH1="./Train.json"  
VAL_PATH="./Val.json"

OUTPUT_DIR="./RemoteReasoner_7B"

NPROC_PER_NODE=8 \
MAX_PIXELS=802816 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift rlhf \
    --rlhf_type grpo \
    --external_plugins ./custom/custom_plugin.py\
    --reward_funcs format external_vg_acc \
    --attn_impl flash_attn \
    --model $MODEL_NAME \
    --train_type lora \
    --split_dataset_ratio 0.0 \
    --ddp_find_unused_parameters false \
    --dataset $DATA_PATH1 \
    --val_dataset $VAL_PATH \
    --torch_dtype bfloat16 \
    --num_train_epochs 24 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --per_device_eval_batch_size $BATCH_PER_DEVICE \
    --learning_rate 1e-6 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --gradient_checkpointing true \
    --eval_steps 40000 \
    --save_steps 10 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir $OUTPUT_DIR \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --num_generations 4 \
    --temperature 0.9 \
    --report_to tensorboard \
    --system ./swift/ms-swift/examples/train/grpo/prompt.txt \
    --deepspeed zero3