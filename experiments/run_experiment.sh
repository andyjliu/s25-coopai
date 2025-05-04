#!/usr/bin/env bash

sim_types=("simulate_via_prompting" "simulate_and_best_response" "simulate_internally" "simulate_externally")
sim_costs=(0.0 0.4 0.8 1.2 1.6)
models=("lambda-deepseek-r1" "lambda-deepseek-v3")
ROUNDS=20
MATRIX=0

cd ../

for MODEL in "${models[@]}"; do
  for SIM_COST in "${sim_costs[@]}"; do
    for SIM_TYPE in "${sim_types[@]}"; do
      echo "Running model=$MODEL sim_cost=$SIM_COST sim_type=$SIM_TYPE"
      uv run src/restricted_trust_with_simulation.py \
        --p1_model "$MODEL" \
        --p2_model "$MODEL" \
        --rounds "$ROUNDS" \
        --simulation_cost "$SIM_COST" \
        --simulation_type "$SIM_TYPE" \
        --csv_output "data/rts_matrix_${MATRIX}/0502_rts_deepseek.csv" \
        --matrix_number "$MATRIX" \
        --verbose \
        --payoff-matrix-path "src/test.json"
    done
  done
done
