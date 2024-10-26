#!/bin/bash

# 定义 alpha 和 beta 的列表
alphas=(0 0.1 0.2)
deltas=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)


# 遍历 alpha 和 beta 的组合进行网格搜索
for alpha in "${alphas[@]}"; do
    for delta in "${deltas[@]}"; do
        echo "Running alpha=${alpha}, delta=${delta} on device=${device_id}"
        python -u main.py --device "0" --M "1000"  --alpha "${alpha}" --delta "${delta}"
    done
done

echo "Grid search completed."
