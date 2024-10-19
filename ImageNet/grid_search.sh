#!/bin/bash

# 候选的alpha、beta值
candidates=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# 结果文件
result_file="grid_search_results.txt"

# 开始进行网格搜索
for alpha in ${candidates[@]}; do
  for beta in ${candidates[@]}; do
    # 计算 delta
    delta=$(echo "1 - $alpha - $beta" | bc)

    # 检查 delta 是否在 0 到 1 的范围内
    if (( $(echo "$delta >= 0" | bc) )) && (( $(echo "$delta <= 1" | bc) )); then
      echo "Running with alpha=$alpha, beta=$beta, delta=$delta"
      
      # 记录当前的alpha, beta, delta组合到结果文件
      echo "alpha=$alpha, beta=$beta, delta=$delta" >> $result_file
      
      # 执行命令
      python -u main.py --device 3 --M 1000 --alpha $alpha --beta $beta --delta $delta

      # 将结果分隔符记录到文件中
      echo "---------------------------------------------------" >> $result_file
    fi
  done
done

echo "Grid search completed."
