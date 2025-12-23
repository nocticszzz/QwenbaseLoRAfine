# 基于 Qwen2.5 + LoRA 的细粒度中文仇恨言论生成式提取

本项目实现了基于大语言模型（LLM）的细粒度中文仇恨言论信息抽取任务。利用 **Qwen2.5-7B-Instruct** 作为基座模型，通过 **LoRA (Low-Rank Adaptation)** 技术进行指令微调，实现了从非结构化社交媒体文本到结构化四元组的端到端生成。
# 1. 创建环境
conda create -n hateful_env python=3.10
conda activate hateful_env

# 2. 安装 PyTorch 
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 3. 安装依赖库
pip install transformers peft datasets
pip install llama-factory  # 使用 LLaMA-Factory 进行微调
pip install bitsandbytes   # 量化支持

llamafactory-cli webui

llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /path/to/Qwen2.5-7B-Instruct \
    --dataset train_data \
    --dataset_dir ./data \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --output_dir ./saves/Qwen2.5-7B/lora/v1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --learning_rate 5e-5 \
    --num_train_epochs 5 \
    --fp16 True

# 使用已经训练好的LoRA checkpoint-400 进行推理 
模型地址：[checkpoint400](https://huggingface.co/nocticszzz/qwenlora/tree/main)  
python predict-fin.py
