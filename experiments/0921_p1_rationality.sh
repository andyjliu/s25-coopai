#!/usr/bin/env bash

# models=( "openrouter-deepseek-r1" "openrouter-deepseek-v3" "gemini-2.5-flash" "openrouter-gpt4.1" "openrouter-gpt4.1-mini" "openrouter-o4-mini" "openrouter-qwen3-next" )
models=( "gemini-2.5-flash")
ROUNDS=9
MATRIX=0

cd ../

for MODEL in "${models[@]}"; do
    uv run src/p1_rationality_test.py \
        --model_name "$MODEL" \
        --n_repeats "$ROUNDS" \
        --csv_path "data/p1_rationality_matrix_${MATRIX}/0921_${MODEL}.csv" \
        --matrix_number "$MATRIX" \
        --payoff_matrix_path "src/test.json" \
        --verbose
    sleep 10
done
