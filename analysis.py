import os

# 指定主目录路径
base_path = os.path.join("/data/hyh/imagenet/data", "train")

# 初始化一个字典来存储每个子目录的图片数量
image_count_per_folder = {}

# 遍历主目录下的所有子目录
for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)
    
    # 仅处理子目录
    if os.path.isdir(folder_path):
        # 统计子目录中的文件数量（假设所有文件都是图片）
        image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        image_count_per_folder[folder_name] = len(image_files)

# 输出每个子目录的图片数量
for folder, count in image_count_per_folder.items():
    print(f"目录 '{folder}' 中有 {count} 张图片。")
