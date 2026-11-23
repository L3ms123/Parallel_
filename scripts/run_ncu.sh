#!/bin/bash
#SBATCH --partition=cuda-ext.q
#SBATCH --gres=gpu:GeForceRTX3080:1
#SBATCH --nodelist=aolin-gpu-1
#SBATCH --output=out_ncu.out
#SBATCH --error=err_ncu.out

module load nvhpc/26.9
hostname
which ncu

tr -d '\r' < cfg_63/cfg_500.txt | sed 's/^num_epochs=10$/num_epochs=1/' > cfg_63/cfg_e1.txt
grep num_epochs cfg_63/cfg_e1.txt

nvc -O3 -acc=gpu -g main.c common/common.c configuration/config.c layer/layer.c \
    randomizer/randomizer.c initialize/initialize.c training/training.c -o exec_gpu -lm

ncu --set full -k regex:forward_prop_156   -s 20 -c 4 -f -o ncu_forward ./exec_gpu cfg_63/cfg_e1.txt
ncu --set full -k regex:back_prop_341      -s 20 -c 3 -f -o ncu_b341    ./exec_gpu cfg_63/cfg_e1.txt
ncu --set full -k regex:back_prop_351      -s 20 -c 2 -f -o ncu_b351    ./exec_gpu cfg_63/cfg_e1.txt
ncu --set full -k regex:update_weights_65  -s 4  -c 4 -f -o ncu_update  ./exec_gpu cfg_63/cfg_e1.txt

for f in ncu_forward ncu_b341 ncu_b351 ncu_update; do
  ncu --import $f.ncu-rep --page details > $f.txt
done
