#!/bin/bash

gradients=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for gradient in "${gradients[@]}"; do
        echo "Running gradient=${gradient} on device=${device_id}"
        python -u main.py --device "2" --M "500" --dnum "4" --gradient "${gradient}"
    done

echo "M=500 shot=4 finished."
