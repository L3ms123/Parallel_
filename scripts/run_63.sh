#!/bin/bash
#SBATCH --partition=cuda-ext.q
#SBATCH --gres=gpu:GeForceRTX3080:1
#SBATCH --nodelist=aolin-gpu-1
#SBATCH --cpus-per-task=1
#SBATCH --output=out_63.out
#SBATCH --error=err_63.out

module load nvhpc/26.9
hostname
nvidia-smi --query-gpu=name --format=csv

ORIG=../nn_original     
FILES="main.c common/common.c configuration/config.c layer/layer.c randomizer/randomizer.c initialize/initialize.c training/training.c"

# GPU (tu código actual)
nvc -O3 -acc=gpu $FILES -o exec_gpu -lm

# CPU (original, sin -acc)
( cd $ORIG && nvc -O3 $FILES -o ../nn_openacc/exec_cpu -lm )

REPS=3
for cfg in cfg_63/cfg_500.txt cfg_63/cfg_1000.txt cfg_63/cfg_2000.txt \
           cfg_63/cfg_b10.txt cfg_63/cfg_b100.txt cfg_63/cfg_b200.txt \
           cfg_63/cfg_e5.txt cfg_63/cfg_e20.txt; do
  for exe in exec_gpu exec_cpu; do
    for r in $(seq $REPS); do
      echo "== $cfg $exe run$r"
      ./$exe $cfg | grep -E "encerts|Goodbye|loss"
    done
  done
done
