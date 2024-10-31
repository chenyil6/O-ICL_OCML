#!/bin/bash

# 定义 alpha 和 beta 的列表
gradients=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)


# 遍历 alpha 和 beta 的组合进行网格搜索
for gradient in "${gradients[@]}"; do
        echo "Running gradient=${gradient} on device=${device_id}"
        python -u main.py --device "1" --M "1000" --dnum "8" --gradient "${gradient}"
    done

echo "M=1k shot=8 finished."
