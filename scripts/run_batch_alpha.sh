#!/bin/bash
#SBATCH --partition=cuda-ext.q
#SBATCH --gres=gpu:GeForceRTX3080:1
#SBATCH --nodelist=aolin-gpu-1
#SBATCH --output=out_balpha.out
#SBATCH --error=err_balpha.out

module load nvhpc/26.9
hostname

for cfg in cfg_63/cfg_b100_a.txt cfg_63/cfg_b200_a.txt; do
  for exe in exec_gpu exec_cpu; do
    for r in 1 2 3; do
      echo "== $cfg $exe run$r"
      ./$exe $cfg | grep -E "encerts|Goodbye"
    done
  done
done
