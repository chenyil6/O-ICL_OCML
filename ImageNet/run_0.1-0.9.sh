#!/bin/bash

beta=0

for alpha in $(seq 0.1 0.1 0.9)
do
    delta=$(echo "1 - ${alpha}" | bc)
    echo "Running with alpha=${alpha} beta=${beta} delta=${delta}"
    python -u main.py --device "3" --M "1000" --alpha "${alpha}" --beta "${beta}" --delta "${delta}"
done
