import os
import shutil
from huggingface_hub import snapshot_download

# 直接指定模型和下载路径
model_name = 'luodian/OTTER-9B-LA-InContext'
download_path = '/data1/chy/OTTER-9B-LA-InContext'

# 确保下载路径存在
os.makedirs(download_path, exist_ok=True)
print('*'*10, f'Start downloading {model_name}', '*'*10)

# 下载模型
snapshot_download(repo_id=model_name,
                  cache_dir=download_path,
                  local_dir=download_path,
                  local_dir_use_symlinks=False,
                  ignore_patterns=["*.h5","*safetensors","*msgpack"],
                  force_download=True,
                  resume_download=False,
                  etag_timeout=60)


print('*'*10, 'Download finish', '*'*10)

# 清理下载的模型目录
for item in os.listdir(download_path):
    if os.path.isdir(os.path.join(download_path, item)) and item.startswith('models'):
        shutil.rmtree(os.path.join(download_path, item))
