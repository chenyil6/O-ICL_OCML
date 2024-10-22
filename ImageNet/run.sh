# 第一条命令
python -u main.py --device "3" --M "1000" --alpha "0.2" --delta "0.8"

# 检查上一条命令是否成功运行
if [ $? -eq 0 ]; then
    echo "First program finished successfully."
else
    echo "First program failed." >&2
    exit 1
fi

# 第二条命令
python -u main.py --device "3" --M "1000" --alpha "0.3" --delta "0.7"

# 检查第二条命令是否成功运行
if [ $? -eq 0 ]; then
    echo "Second program finished successfully."
else
    echo "Second program failed." >&2
    exit 1
fi

# 第二条命令
python -u main.py --device "3" --M "1000" --alpha "0.4" --delta "0.6"

# 检查第二条命令是否成功运行
if [ $? -eq 0 ]; then
    echo "Third program finished successfully."
else
    echo "Third program failed." >&2
    exit 1
fi

# 第二条命令
python -u main.py --device "3" --M "1000" --alpha "0.6" --delta "0.4"

# 检查第二条命令是否成功运行
if [ $? -eq 0 ]; then
    echo "Fourth program finished successfully."
else
    echo "Fourth program failed." >&2
    exit 1
fi

python -u main.py --device "3" --M "1000" --alpha "0.7" --delta "0.3"

# 检查第二条命令是否成功运行
if [ $? -eq 0 ]; then
    echo "Fifth program finished successfully."
else
    echo "Fifth program failed." >&2
    exit 1
fi

python -u main.py --device "3" --M "1000" --alpha "0.8" --delta "0.2"

# 检查第二条命令是否成功运行
if [ $? -eq 0 ]; then
    echo "Sixth program finished successfully."
else
    echo "Sixth program failed." >&2
    exit 1
fi

python -u main.py --device "3" --M "1000" --alpha "0.9" --delta "0.1"

# 检查第二条命令是否成功运行
if [ $? -eq 0 ]; then
    echo "Seventh program finished successfully."
else
    echo "Seventh program failed." >&2
    exit 1
fi

