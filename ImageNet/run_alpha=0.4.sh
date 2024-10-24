#!/bin/bash

alpha=0.4
min_beta=0
max_beta=$(echo "1 - ${alpha}" | bc)

for beta in $(seq "${min_beta}" 0.1 "${max_beta}")
do
    delta=$(echo "1 - ${alpha} - ${beta}" | bc)
    echo "Running with alpha=${alpha}, beta=${beta}, delta=${delta}"
    python -u main.py --device "3" --method "Online_ICL" --M "1000" --alpha "${alpha}" --beta "${beta}" --delta "${delta}"
done
