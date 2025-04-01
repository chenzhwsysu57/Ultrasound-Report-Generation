python KMVE_RG/my_main.py --dataset_name all --model moe --batch_size 64 --num_experts 3 --num_shared_experts 1 --d_ff 1024 --seed 46 --result moe-share1route3-dff1024-bs64-seed46 --num_workers 10

python KMVE_RG/my_main.py --dataset_name all --model moe --batch_size 64 --num_experts 6 --num_shared_experts 1 --d_ff 1024 --seed 42 --result moe-share1route6-dff1024-bs64-seed42 --num_workers 10
python KMVE_RG/my_main.py --dataset_name all --model moe --batch_size 64 --num_experts 6 --num_shared_experts 1 --d_ff 1024 --seed 43 --result moe-share1route6-dff1024-bs64-seed43 --num_workers 10
python KMVE_RG/my_main.py --dataset_name all --model moe --batch_size 64 --num_experts 6 --num_shared_experts 1 --d_ff 1024 --seed 44 --result moe-share1route6-dff1024-bs64-seed44 --num_workers 10