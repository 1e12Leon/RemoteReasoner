import sys
import json
import re
import traceback
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from utils.launch_utils import launch
from utils.comm import get_rank, get_world_size, get_local_rank
from swift.llm import InferEngine, InferRequest, VllmEngine, PtEngine, RequestConfig, get_template, \
    safe_snapshot_download, \
    BaseArguments
from swift.tuners import Swift
import asyncio
import torch


def gen_model(args, device=0):
    """Load model to specified device"""
    adapter_path = safe_snapshot_download(args.lora_path)
    args_info = BaseArguments.from_pretrained(adapter_path)
    args_info.device_map = device
    logger.info(f"Rank {get_rank()} loading model to device {device}")
    model, processor = args_info.get_model_processor()
    model = Swift.from_pretrained(model, adapter_path)
    template = args_info.get_template(processor)
    engine = PtEngine.from_model_template(model, template)
    return engine, processor

def main(args):
    # ========== Initialize distributed environment ==========
    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()
    logger.info(f"Start inference process [RANK {rank}/{world_size}] [LOCAL_RANK {local_rank}]")

    # ========== Each process loads its own model copy ==========
    engine, processor = gen_model(args, device=local_rank)
    
    # ========== Load data and global sharding ==========
    with open(args.json_file) as jf:
        full_data = json.load(jf)
    
    # Global sharding (considering multi-machine)
    chunk_size = len(full_data) // world_size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size if rank < world_size - 1 else len(full_data)
    local_data = full_data[start_idx:end_idx]
    
    logger.info(f"Process {rank} handles {len(local_data)} samples (index {start_idx}-{end_idx})")

    # ========== Create save directory ==========
    os.makedirs(args.save_json_dir, exist_ok=True)
    
    # ========== Batch inference ==========
    request_config = RequestConfig()
    
    indices = range(0, len(local_data), args.max_batch)
    progress_bar = tqdm(
        indices,
        desc=f"Rank {rank} inference progress",
        position=rank,
        disable=(rank != 0)
    )
    for batch_start in progress_bar:
        batch_end = min(batch_start + args.max_batch, len(local_data))
        batch_data = local_data[batch_start:batch_end]
        
        try:
            # Prepare ground truth text (according to RL flag)
            if args.rl:
                gt_texts = [infer_data['solution'] for infer_data in batch_data]
            else:
                gt_texts = [InferRequest.remove_response(infer_data['messages']) for infer_data in batch_data]

            # Synchronous inference
            resp_list = engine.infer(batch_data, request_config)
            
            # Process each response
            for idx, (response, gt_text) in enumerate(zip(resp_list, gt_texts)):
                infer_text = ""
                think = ""
                
                # Special handling for RL mode
                if args.rl:
                    match = re.match(
                        r'^<think>(.*?)</think>\s*<answer>(.*?)</answer>$',
                        response.choices[0].message.content,
                        re.DOTALL | re.MULTILINE
                    )
                    if match:
                        think = match.group(1).strip()
                        infer_text = match.group(2).strip()
                    else:
                        logger.warning(f"Rank {rank} response format mismatch: {response.choices[0].message.content}")
                        infer_text = response.choices[0].message.content
                else:
                    infer_text = response.choices[0].message.content
                
                # Build result
                result = {
                    "infer_text": infer_text,
                    "gt_text": gt_text,
                    "think": think,
                    "raw_data": batch_data[idx],
                    "rank": rank,
                    "batch_index": batch_start + idx
                }
                print(infer_text)
                print(result["raw_data"]['images'])
                # Safe file write (avoid conflicts)
                img_name = os.path.basename(result["raw_data"]['images'])
                save_file = os.path.join(
                    args.save_json_dir, 
                    f"rank{rank}_batch{batch_start}_item{idx}_{img_name[:-4]}.json"
                )
                with open(save_file, 'w') as sf:
                    json.dump(result, sf, indent=4)
                    
        except Exception as e:
            logger.error(f"Rank {rank} batch {batch_start} inference failed")
            logger.error(traceback.format_exc())
            
            # Reduce batch size if out of memory
            if "out of memory" in str(e).lower() and args.max_batch > 1:
                new_batch_size = max(1, args.max_batch // 2)
                logger.warning(f"Rank {rank} out of memory, reduce batch_size {args.max_batch} -> {new_batch_size}")
                args.max_batch = new_batch_size
    
    # Release resources
    logger.info(f"Rank {rank} inference finished, releasing resources")
    del engine
    torch.cuda.empty_cache()

if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str,
                        default="/disk/deepdata/dataset/RemoteSAM-270K/test/RefSegRS_R1.json")
    parser.add_argument('--save_json_dir', type=str,
                        default="/disk/deepdata/dataset/RemoteSAM-270K/RemoteSAM_R1/test_RefSegRS/")
    parser.add_argument('--lora_path', type=str,
                        default="/disk/deepdata/dataset//RemoteSAM-270K/RemoteSAM_R1/v11-20250611-181949/checkpoint-2600/") 
    parser.add_argument('--num_gpus', type=int, default=8,
                        help="Number of GPUs per machine")
    parser.add_argument("--max_batch", type=int, default=4,
                        help="Maximum batch size")
    parser.add_argument("--num_machines", type=int, default=4,
                        help="Number of machines")
    parser.add_argument("--machine_rank", type=int, default=0,
                        help="Current machine rank")
    parser.add_argument("--dist_url", type=str, default="tcp://127.0.0.1:25969",
                        help="Distributed connection URL")
    parser.add_argument("--rl", type=int, default=1,
                        help="RL mode (0/1)")
    
    args = parser.parse_args()

    
    # Single GPU mode
    if args.num_gpus == 1:
        logger.info("Single GPU mode start")
        main(args)
    # Multi-GPU mode
    else:
        logger.info(f"Distributed start: {args.num_gpus} GPUs, {args.num_machines} machines")
        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
