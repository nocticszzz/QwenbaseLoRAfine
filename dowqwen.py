from modelscope import snapshot_download

# 指定下载目录
save_dir = "/home/tongzy/models/Qwen2.5-7B-Instruct"

print(f"开始下载模型到: {save_dir} ...")
model_dir = snapshot_download(
    'qwen/Qwen2.5-7B-Instruct', 
    cache_dir=None, 
    local_dir=save_dir
)
print(f"下载完成！模型路径为: {model_dir}")