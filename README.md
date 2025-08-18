<div align="center">

# [RemoteReasoner: Towards Unifying Geospatial Reasoning Workflow](https://arxiv.org/pdf/2507.19280)



[Liang Yao (姚亮)](https://1e12leon.top/) 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp; 
[Fan Liu (刘凡)](https://multimodality.group/author/%E5%88%98%E5%87%A1/) ✉ 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp;
Hongbo Lu (陆泓波)
<img src="assets/COWA.jpg" alt="Logo" width="15">, &nbsp; &nbsp;
[Chuanyi Zhang (张传一)](https://ai.hhu.edu.cn/2023/0809/c17670a264073/page.htm) 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp;

[Rui Min (闵锐)]() 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp;
[Shengxiang Xu (徐圣翔)](xushengxianggg.github.io) 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp;
[Shimin Di (邸世民)](https://cs.seu.edu.cn/shimindi/main.htm) 
<img src="assets/SEU.png" alt="Logo" width="15">, &nbsp; &nbsp; 
[Pai Peng (彭湃)](https://scholar.google.com/citations?user=s8m-hZoAAAAJ&hl=zh-CN) 
<img src="assets/COWA.jpg" alt="Logo" width="15">

\*  ✉ *Corresponding Author*

🤗 Model: [RemoteReasoner](https://huggingface.co/1e12Leon/RemoteReasoner)

</div>

## News

- **2025/08/16** Welcome to RemoteReasoner. This is the first Reinforcement Learning-based reasoning framework in remote sensing.


## Introduction

Remote sensing imagery presents vast, inherently unstructured spatial data, necessitating sophisticated reasoning to interpret complex user intents and contextual relationships beyond simple recognition tasks. In this paper, we aim to construct an Earth observation workflow to handle complex queries by reasoning about spatial context and user intent. As a reasoning workflow, it should autonomously explore and construct its own inference paths, rather than being confined to predefined ground‑truth sequences.
Ideally, its architecture ought to be unified yet generalized, possessing capabilities to perform diverse reasoning tasks through one model without requiring additional fine-tuning.
Existing remote sensing approaches rely on supervised fine-tuning paradigms and task‑specific heads, limiting both autonomous reasoning and unified generalization.
To this end, we propose RemoteReasoner, a unified workflow for geospatial reasoning. The design of RemoteReasoner integrates a multi-modal large language model (MLLM) for interpreting user instructions and localizing targets, together with task transformation strategies that enable multi-granularity tasks, including object-, region-, and pixel-level. 
In contrast to existing methods, our framework is trained with reinforcement learning (RL) to endow the MLLM sufficient reasoning autonomy. 
At the inference stage, our transformation strategies enable diverse task output formats without requiring task-specific decoders or further fine-tuning. Experiments demonstrated that RemoteReasoner achieves state-of-the-art (SOTA) performance across multi-granularity reasoning tasks. Furthermore, it retains the MLLM's inherent generalization capability, demonstrating robust performance on unseen tasks and out-of-distribution categories. 
![RemoteReasoner](assets/RemoteReasoner.jpg)


## Quick Start

### Setting Up

1. Clone this repository.
2. Change directory to root of this repository.
3. Create a new environment and install the packages via pip.

```shell
git clone https://github.com/1e12Leon/RemoteReasoner.git
cd RemoteReasoner
pip install -e .
```

Download the pre-trained weights of RemoteReasoner from [HuggingFace](https://huggingface.co/1e12Leon/RemoteReasoner)


### Training
We provides training scripts to fine-tune Qwen2.5-VL-7B-Instruct with GRPO (Group Relative Policy Optimization) using LoRA. The training leverages multi-GPU distributed training with DeepSpeed ZeRO-3 for efficient memory usage.

```shell
bash RemoteReasoner_GRPO.sh
```
#### ⚙️ Key Arguments

| Category              | Argument                              | Description                                                                 | Default / Value |
|------------------------|---------------------------------------|-----------------------------------------------------------------------------|-----------------|
| **Model & Dataset**    | `--model`                            | Path to the base model (e.g., Qwen2.5-VL-7B-Instruct).                      | `./Qwen2.5-VL/Qwen2.5-VL-7B-Instruct/` |
|                        | `--dataset`                          | Path to the training dataset.                                               | `./Train.json`  |
|                        | `--val_dataset`                      | Path to the validation dataset.                                             | `./Val.json`    |
| **Training Config**    | `--rlhf_type`                        | Reinforcement learning type.                                                | `grpo`          |
|                        | `--train_type`                       | Training method (LoRA fine-tuning).                                         | `lora`          |
|                        | `--torch_dtype`                      | Data type for training.                                                     | `bfloat16`      |
|                        | `--num_train_epochs`                 | Number of training epochs.                                                  | `24`            |
|                        | `--learning_rate`                    | Learning rate.                                                              | `1e-6`          |
| **LoRA Parameters**    | `--lora_rank`                        | Rank of the LoRA decomposition.                                             | `8`             |
|                        | `--lora_alpha`                       | Scaling factor for LoRA adaptation.                                         | `16`            |
|                        | `--target_modules`                   | Apply LoRA to specific modules.                                             | `all-linear`    |
| **Batch & Optimizer**  | `--per_device_train_batch_size`       | Batch size per GPU.                                                         | `8`             |
|                        | `--gradient_accumulation_steps`      | Steps to accumulate gradients before update.                                | `8`             |
|                        | `--gradient_checkpointing`           | Enable memory-efficient gradient checkpointing.                             | `true`          |
|                        | `--warmup_ratio`                     | Ratio of total steps for LR warmup.                                         | `0.05`          |
| **Eval & Logging**     | `--eval_steps`                       | Run evaluation every N steps.                                               | `40000`         |
|                        | `--save_steps`                       | Save checkpoints every N steps.                                             | `10`            |
|                        | `--save_total_limit`                 | Keep only the most recent checkpoints.                                      | `2`             |
|                        | `--logging_steps`                    | Log training metrics every N steps.                                         | `5`             |
|                        | `--report_to`                        | Logging backend.                                                            | `tensorboard`   |
| **Generation & Reward**| `--num_generations`                  | Number of generations per step.                                             | `4`             |
|                        | `--temperature`                      | Sampling temperature for text generation.                                   | `0.9`           |
|                        | `--reward_funcs`                     | Reward functions used (e.g., format, external visual grounding accuracy).    | `format external_vg_acc` |
|                        | `--external_plugins`                 | Path to custom plugin for external rewards.                                 | `./custom/custom_plugin.py` |
| **Distributed**        | `--deepspeed`                        | Enable DeepSpeed optimization.                                              | `zero3`         |
|                        | `--ddp_find_unused_parameters`       | Whether to allow unused parameters in DDP.                                  | `false`         |
|                        | `NPROC_PER_NODE`                     | Number of processes (GPUs) per node.                                        | `8`             |
|                        | `CUDA_VISIBLE_DEVICES`               | List of GPUs used for training.                                             | `0,1,2,3,4,5,6,7` |

### Getting Started 

Firstly, you need initialize a model and load the RemoteReasoner checkpoint with a few lines of code:

```python
import cv2
import numpy as np
from RemoteReasoner import RemoteReasoner

parser = argparse.ArgumentParser()
parser.add_argument('--lora_path', type=str, required=True, 
                    help="Path to the LoRA adapter model")
args = parser.parse_args()

logger.info("Initializing RemoteReasoner...")
reasoner = RemoteReasoner(args, device=0)
    
```

- **Pixel Reasoning**

```python
img_path = "./assets/demo.jpg"
queston = "your query."
think, answer, mask = reasoner.Pixel_reasoning(img_path, question)
```


- **Region Reasoning**

```python
img_path = "./assets/demo.jpg"
queston = "your query."
think, answer = reasoner.Region_reasoning(img_path, question)
```

- **Contour Reasoning**

```python
img_path = "./assets/demo.jpg"
queston = "your query."
think, answer, contour = reasoner.Contour_reasoning(img_path, question)
```

- **Visual Queston Answering**

```python
img_path = "./assets/demo.jpg"
queston = "your question."
think, answer = reasoner.VQA(img_path, question)
```

- **Image Captioning**

```python
img_path = "./assets/demo.jpg"
think, answer, mask = reasoner.Image_captioning(img_path)
```

## Acknowledge

- Thanks to [Kaiyu](https://likyoo.github.io/) for providing the [EarthReason](https://huggingface.co/datasets/earth-insights/EarthReason) dataset
- Thanks for the [MS-SWIFT](https://github.com/modelscope/ms-swift.git) repo.

## Cite
If you find this work useful, please cite our paper as:

```
@article{yao2025remotereasoner,
  title={RemoteReasoner: Towards Unifying Geospatial Reasoning Workflow},
  author={Yao, Liang and Liu, Fan and Lu, Hongbo and Zhang, Chuanyi and Min, Rui and Xu, Shengxiang and Di, Shimin and Peng, Pai},
  journal={arXiv preprint arXiv:2507.19280},
  year={2025}
}
