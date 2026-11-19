#!/bin/bash
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
## From PC labs: cuda-int; otherwise, use cuda-ext.q
#SBATCH --partition=cuda-ext.q
## GPUs are available in aolin cluster
#SBATCH --gres=gpu:GeForceRTX3080:1
#SBATCH -o out.out
#SBATCH -e err.out

#You can run this with: "salloc -p interactive.q -w aolin-gpu-2 --gres=shard:1 -N 1", "chmod +x run.sh" and "./run.sh"

set -e
module load nvhpc/26.9
nvc -O3 -acc=gpu -Minfo=all main.c common/common.c configuration/config.c layer/layer.c randomizer/randomizer.c initialize/initialize.c training/training.c -o exec
./exec

# Profiling:
#nsys nvprof --print-gpu-trace ./exec #summary 

# Detailed Profiling:
#ncu --target-processes application-only --set full -f -o profile.ncu-rep ./exec

# 1. Manually download file "profile.ncu-rep" to your computer
# 2. Download the "Nsys" profiler from the Nvidia website on your computer (https://developer.nvidia.com/nsight-systems).
# 3. Use "ncu" from your computer to open "profile.ncu-rep"

# Profile traces with nsys:
# nsys profile -f true -t nvtx,cuda -o profile.nsys-rep ./main 100 100
# Download "profile.nsys-rep.qdrep" on your computer and use nsys to open the file.