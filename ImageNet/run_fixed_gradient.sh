# 第一条命令
python -u main.py --device "0" --update_strategy "fixed_gradient" --M "1000" --alpha "0" --beta "0" --delta "0" --temperature "3" --gradient "0.3"

# 检查上一条命令是否成功运行
if [ $? -eq 0 ]; then
    echo "First program finished successfully."
else
    echo "First program failed." >&2
    exit 1
fi

# 第二条命令
python -u main.py --device "0" --update_strategy "fixed_gradient" --M "1000" --alpha "0" --beta "0" --delta "0" --temperature "3" --gradient "0.4"

# 检查第二条命令是否成功运行
if [ $? -eq 0 ]; then
    echo "Second program finished successfully."
else
    echo "Second program failed." >&2
    exit 1
fi

# 第二条命令
python -u main.py --device "0" --update_strategy "fixed_gradient" --M "1000" --alpha "0" --beta "0" --delta "0" --temperature "3" --gradient "0.6"


# 检查第二条命令是否成功运行
if [ $? -eq 0 ]; then
    echo "Third program finished successfully."
else
    echo "Third program failed." >&2
    exit 1
fi

# 第二条命令
python -u main.py --device "0" --update_strategy "fixed_gradient" --M "1000" --alpha "0" --beta "0" --delta "0" --temperature "3" --gradient "0.7"

# 检查第二条命令是否成功运行
if [ $? -eq 0 ]; then
    echo "Fourth program finished successfully."
else
    echo "Fourth program failed." >&2
    exit 1
fi

python -u main.py --device "0" --update_strategy "fixed_gradient" --M "1000" --alpha "0" --beta "0" --delta "0" --temperature "3" --gradient "0.8"


# 检查第二条命令是否成功运行
if [ $? -eq 0 ]; then
    echo "Fifth program finished successfully."
else
    echo "Fifth program failed." >&2
    exit 1
fi

python -u main.py --device "0" --update_strategy "fixed_gradient" --M "1000" --alpha "0" --beta "0" --delta "0" --temperature "3" --gradient "0.9"

# 检查第二条命令是否成功运行
if [ $? -eq 0 ]; then
    echo "Sixth program finished successfully."
else
    echo "Sixth program failed." >&2
    exit 1
fi


