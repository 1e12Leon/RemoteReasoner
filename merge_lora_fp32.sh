swift export \
    --adapters checkpoints \
    --merge_lora true \
    --torch_dtype float32 \
    --output_dir checkpoints/RemoteReasoner-7B-merged-fp32 \
    --external_plugins "" 