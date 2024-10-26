#!/bin/bash

# 指定你的GPU设备编号，例如：(0 1 2 3)
devices=(0 1 2 3)

# 超参数的值，你可以根据需要修改
alphas=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
betas=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
deltas=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

device_id=0

for alpha in "${alphas[@]}"; do
  for beta in "${betas[@]}"; do
    for delta in "${deltas[@]}"; do
      # 运行你的程序，并指定GPU设备和超参数
      python main.py --model "open_flamingo" --method "Online_ICL" --update_strategy "gradient_prototype" --alpha $alpha --beta $beta --delta $delta --device ${devices[$device_id]} &

      # 更新设备编号，如果超过了设备数组的长度，就从头开始
      device_id=$((device_id+1))
      if [ $device_id -ge ${#devices[@]} ]; then
        device_id=0
      fi

      # 等待所有的GPU设备都完成任务
      if [ $device_id -eq 0 ]; then
        wait
      fi
    done
  done
done
