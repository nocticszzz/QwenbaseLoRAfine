import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

#配置区域
BASE_MODEL_PATH = "Qwen模型所在的路径"
# 请确认这个路径是否正确
LORA_PATH = "saves/Qwen2.5-7B-Instruct/lora/train_2025-12-19-20-17-18/checkpoint-400"
TEST_FILE = "test1.json"
OUTPUT_FILE = "demo.txt"

# 2. Batch Size 
BATCH_SIZE = 4 

SYSTEM_PROMPT = (
    "你是一个网络安全领域的专家，专注于细粒度中文仇恨言论识别任务。\n"
    "任务说明：\n"
    "1. 请分析给定的社交媒体文本，提取出所有的仇恨或非仇恨四元组。\n"
    "2. 四元组结构为：评论对象(Target) | 论点(Argument) | 目标群体(Targeted Group) | 是否仇恨(Hateful)。\n"
    "3. 格式严格要求：\n"
    "   - 每个四元组内部使用 ' | ' (空格+竖线+空格) 分割。\n"
    "   - 如果存在多个四元组，使用 ' [SEP] ' 分割。\n"
    "   - 整个回复必须以 ' [END]' 结尾。\n"
    "   - 对于没有特定对象的实例，Target设为NULL。\n"
    "   - 目标群体包括：地域、种族、性别、LGBTQ、其他、non-hate。\n"
    "   - 是否仇恨标签为：hate 或 non-hate。\n"
    "4. 请直接输出格式化后的结果，不要包含其他解释性文字。"
)

def main():
    print("正在初始化...")
    
    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # 加载模型 
    print("加载基座模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16, # 使用 float16
        trust_remote_code=True
    ).cuda() # 显式移动到 CUDA

    print(f"加载 LoRA: {LORA_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    # 读取数据
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except:
            f.seek(0)
            data = [json.loads(line) for line in f]
    
    all_texts = [item.get('content') or item.get('text') for item in data]
    total_len = len(all_texts)
    print(f"开始推理: 总数 {total_len}, Batch Size {BATCH_SIZE}")

    results = []
    
    # 批处理循环
    for i in range(0, total_len, BATCH_SIZE):
        batch_texts = all_texts[i : i + BATCH_SIZE]
        
        # 构造 Prompt
        batch_prompts = []
        for text in batch_texts:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            batch_prompts.append(prompt)
            
        # Tokenize
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=2048
        ).to(model.device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
            
        # 解码
        generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
        batch_results = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        results.extend([res.strip() for res in batch_results])

        # 打印进度
        print(f"[{min(i + BATCH_SIZE, total_len)}/{total_len}] -> {batch_results[0][:20]}...")

    # 保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(res + '\n')
    print("完成！")

if __name__ == "__main__":
    main()