swift export \
    --adapters checkpoints \
    --merge_lora true \
    --torch_dtype bfloat16 \
    --output_dir checkpoints/RemoteReasoner-7B-merged-bf16 \
    --external_plugins "" 