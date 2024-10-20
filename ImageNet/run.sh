#!/bin/bash
# 第一条命令
python -u main.py --device "3" --M "1000" --alpha "1.0" --beta "1.0" --delta "0"

# 检查上一条命令是否成功运行
if [ $? -eq 0 ]; then
    echo "First program finished successfully."
else
    echo "First program failed." >&2
    exit 1
fi

# 第一条命令
python -u main.py --device "3" --M "1000" --alpha "0" --beta "1.0" --delta "0"

# 检查上一条命令是否成功运行
if [ $? -eq 0 ]; then
    echo "First program finished successfully."
else
    echo "First program failed." >&2
    exit 1
fi

# 第二条命令
python -u main.py --device "3" --M "1000" --alpha "0" --beta "0" --delta "1.0"

# 检查第二条命令是否成功运行
if [ $? -eq 0 ]; then
    echo "Second program finished successfully."
else
    echo "Second program failed." >&2
    exit 1
fi
