#!/usr/bin/env bash

models=( "gemini-2.5-flash" )
ROUNDS=1
MATRIX=0

cd ../

for MODEL in "${models[@]}"; do
    uv run src/p1_rationality_test.py \
        --model_name "$MODEL" \
        --n_repeats "$ROUNDS" \
        --csv_path "data/p1_rationality_matrix_${MATRIX}/0921_${MODEL}.csv" \
        --matrix_number "$MATRIX" \
        --payoff_matrix_path "src/test.json"
done