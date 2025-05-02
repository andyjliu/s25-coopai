#!/bin/bash
#SBATCH --job-name=trust_game
#SBATCH --output=/home/andyliu/s25-coopai/slurm_logs/0501/0501_trust_game_%A_%a.out
#SBATCH --time=1-00:00:00
#SBATCH --mem=16G
#SBATCH --partition=array
#SBATCH --array=1-60

source /home/andyliu/Miniconda3/etc/profile.d/conda.sh
export CONDA_ALWAYS_YES="true"
conda activate coopai
cd /home/andyliu/s25-coopai/src
sim_costs=(0.0 0.4 0.8 1.2 1.6)
models=("gpt-4.1" "gpt-4.1-mini" "o4-mini")
simulation_methods=("simulate_and_best_response" "simulate_via_prompting" "simulate_internally" "simulate_externally")

index=$((SLURM_ARRAY_TASK_ID - 1))
model_index=$((index % 3))
simulation_method_index=$((index % 4))
sim_cost_index=$((index % 5))

python restricted_trust_with_simulation.py \
  --p1_model ${models[$model_index]} \
  --p2_model ${models[$model_index]} \
  --rounds 20 \
  --simulation_cost ${sim_costs[$sim_cost_index]} \
  --payoff-matrix-path test.json \
  --simulation_type ${simulation_methods[$simulation_method_index]} \
  --csv_output /home/andyliu/s25-coopai/data/0501_rts_oai.csv
