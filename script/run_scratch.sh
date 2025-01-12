GPU_ID=1

CUDA_VISIBLE_DEVIC  ES=$GPU_ID python main.py --config cfgs/finetune_scan_hardest.yaml --scratch_model --exp_name Mamba3D_hardest_scratch