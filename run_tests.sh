#!/bin/bash

for seed in {1000..5000}; do
    python KMVE_RG/test_metric.py --model MoEModel --method moe --comment moe --num_experts 3 --num_shared_experts 1 --d_ff 1024 --ckpt /home/chenzhw/ultrasound_report_gen/Nassir-US-Report-Gen/Result/moeloss-share1route3-dff1024-bs64-seed46/Models/all_epoch_89_checkpoint.pth --seed $seed
done

