#!/bin/bash

# List of scales to process
scales=(0.01 0.02 0.04 0.06 0.08 0.09)
max_jobs=2

for scale in "${scales[@]}"; do
    # Run the command in background
    python task_LoRaANN_kaiming_uniform.py --train_bp True --gpu True --path "/mnt/exDisk0/git_repo/Para-HADES/LoRa/W0 fixed - kaiming_uniform" --base_scale "$scale" &

    # If we hit the limit, wait for one to finish before starting the next
    while [ $(jobs -rp | wc -l) -ge $max_jobs ]; do
        sleep 1
    done
done

# Ensure all remaining tasks finish before the script exits
wait
echo "All tasks complete."