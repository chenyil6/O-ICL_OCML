# 设定参数集合
dataset_modes=("balanced")
update_strategies=("minMargin" "meanMargin")


# 遍历参数集合
for dataset_mode in "${dataset_modes[@]}"; do
  for update_strategy in "${update_strategies[@]}"; do
    # 运行 Python 程序
    python -u main.py --device "3" --M "1000" --dataset_mode "$dataset_mode" --update_strategy "$update_strategy" 
  done
done