#!/bin/bash

# 循环指定 ASCEND_RT_VISIBLE_DEVICES 从 0 到 7
for device_id in {0..7}
do
    # 每个设备启动一个 Python 进程，指定 ASCEND_RT_VISIBLE_DEVICES 并在后台运行
    ASCEND_RT_VISIBLE_DEVICES=$device_id python main_npu.py &
done

# 等待所有后台任务完成
wait
echo "All processes are complete."
