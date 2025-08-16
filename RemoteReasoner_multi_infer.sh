CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python infer.py \
  --json_file /disk/deepdata/dataset/RemoteReason/SFT_data/EarthReason_test_SFT.json \
  --save_json_dir /disk/deepdata/dataset/RemoteReason/infer_test_7B_VG_final_SFT/ \
  --lora_path /disk/deepdata/dataset/RemoteReason/RemoteReasoner_SFT_7B/v1-20250804-171700/checkpoint-333 \
  --num_gpus 8 \
  --num_machines 1 \
  --machine_rank 0 \
  --max_batch 8 \
  --rl 1