#!/bin/bash


############
# Usage
############


####################################
# ZINC - 4 SEED RUNS OF EACH EXPTS
####################################

seed0=41
seed1=95
seed2=12
seed3=35
code=main_ZINC_graph_regression.py 
dataset=ZINC

CUDA_AVAILABLE_DEVICES=2 python $code --dataset $dataset --gpu_id 2 --seed $seed0 --config 'configs/neurips/Transformer_ZINC_4_16pLapPE_signinv_GIN_masked.json'
CUDA_AVAILABLE_DEVICES=2 python $code --dataset $dataset --gpu_id 2 --seed $seed1 --config 'configs/neurips/Transformer_ZINC_4_16pLapPE_signinv_GIN_masked.json' 
CUDA_AVAILABLE_DEVICES=2 python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/neurips/Transformer_ZINC_4_16pLapPE_signinv_GIN_masked.json'
CUDA_AVAILABLE_DEVICES=2 python $code --dataset $dataset --gpu_id 2 --seed $seed3 --config 'configs/neurips/Transformer_ZINC_4_16pLapPE_signinv_GIN_masked.json' 
