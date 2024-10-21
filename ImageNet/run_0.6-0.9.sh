#!/bin/bash

# 遍历 alpha 从 0.1 到 0.9，每次递增 0.1
for alpha in $(seq 0.6 0.1 0.9)
do
    echo "Running with alpha=${alpha}"
    python -u main.py --device "2" --M "1000" --alpha "${alpha}"
done
