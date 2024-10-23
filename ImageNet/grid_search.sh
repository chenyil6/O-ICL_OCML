#!/bin/bash

for alpha in $(seq 0.1 0.1 0.9)
do
    for beta in $(seq 0.1 0.1 $(echo "1 - ${alpha}" | bc))  # 确保 alpha + beta <= 1
    do
        delta=$(echo "1 - ${alpha} - ${beta}" | bc)
        echo "Running with alpha=${alpha} beta=${beta} delta=${delta}"
        python -u main.py --device "2" --M "1000" --alpha "${alpha}" --beta "${beta}" --delta "${delta}"
    done
done
