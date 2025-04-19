import os
import shutil
import re

# 设置目录路径
base_dir = "orz_ckpt/debug_orz_7b_ppo_needlebench_mix_math"

# 获取所有iter文件夹
iter_dirs = []
for item in os.listdir(base_dir):
    if os.path.isdir(os.path.join(base_dir, item)) and item.startswith("iter"):
        iter_dirs.append(item)

# 解析迭代次数并排序
iter_numbers = []
for dir_name in iter_dirs:
    match = re.search(r'iter(\d+)', dir_name)
    if match:
        iter_number = int(match.group(1))
        iter_numbers.append((dir_name, iter_number))

iter_numbers.sort(key=lambda x: x[1])

# 确定要保留的迭代次数（500的倍数）
to_keep = []
for dir_name, iter_num in iter_numbers:
    if iter_num % 500 == 0 or iter_num == 50:  # 保留500的倍数和第一个checkpoint
        to_keep.append(dir_name)

# 确定要删除的文件夹
to_delete = [dir_name for dir_name, _ in iter_numbers if dir_name not in to_keep]

# 显示要删除的文件夹
print("将要删除以下文件夹:")
for dir_name in to_delete:
    full_path = os.path.join(base_dir, dir_name)
    size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
               for dirpath, _, filenames in os.walk(full_path) 
               for filename in filenames)
    print(f"{full_path} (大约 {size / (1024**3):.2f} GB)")

print("\n将要保留以下文件夹:")
for dir_name in to_keep:
    print(os.path.join(base_dir, dir_name))

# 询问确认
confirm = input("\n确认删除这些文件夹? (y/n): ")

if confirm.lower() == 'y':
    for dir_name in to_delete:
        full_path = os.path.join(base_dir, dir_name)
        print(f"正在删除 {full_path}...")
        shutil.rmtree(full_path)
    print("删除完成!")
else:
    print("操作已取消")