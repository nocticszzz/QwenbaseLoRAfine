import json
import random
import os

# 输入文件名 (请根据实际情况修改)
INPUT_FILE = 'train.json' 

# 输出文件名
OUTPUT_TRAIN_FILE = 'hateful_sft_train.json'
OUTPUT_VAL_FILE = 'hateful_sft_val.json'

# 验证集比例 (推荐 0.1，即 10% 的数据用于验证效果)
VAL_RATIO = 0.1 

# 随机种子 (保证每次切分结果一致)
SEED = 42

# 系统指令 (System Instruction)
# 根据 mession.txt 的要求定制，明确四元组结构和分隔符
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

def preprocess():
    # 1. 检查文件是否存在
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到文件 {INPUT_FILE}，请确认文件名是否正确。")
        return

    print(f"正在读取 {INPUT_FILE} ...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_count = len(data)
    print(f"成功读取 {total_count} 条原始数据。")

    # 2. 格式转换
    formatted_data = []
    for item in data:
        # 获取原文和目标输出
        content = item.get('content', '')
        target_output = item.get('output', '')
        
        # 简单的数据清洗（防止空数据）
        if not content or not target_output:
            continue
            
        # 构造 LLaMA-Factory (Alpaca) 格式
        formatted_entry = {
            "instruction": SYSTEM_PROMPT,
            "input": content,
            "output": target_output
        }
        formatted_data.append(formatted_entry)

    # 3. 打乱数据并切分验证集
    random.seed(SEED)
    random.shuffle(formatted_data)
    
    val_size = int(len(formatted_data) * VAL_RATIO)
    train_data = formatted_data[val_size:]
    val_data = formatted_data[:val_size]

    # 4. 保存文件
    print(f"正在保存...")
    
    with open(OUTPUT_TRAIN_FILE, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"   - 训练集 ({len(train_data)} 条) 已保存至: {OUTPUT_TRAIN_FILE}")
    
    with open(OUTPUT_VAL_FILE, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    print(f"   - 验证集 ({len(val_data)} 条) 已保存至: {OUTPUT_VAL_FILE}")

    print("\n预处理完成！")

if __name__ == '__main__':
    preprocess()