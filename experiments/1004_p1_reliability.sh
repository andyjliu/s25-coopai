#!/usr/bin/env bash

models=( "openrouter-deepseek-r1" "openrouter-deepseek-v3" "gemini-2.5-flash" "openrouter-gpt4.1" "openrouter-gpt4.1-mini" "openrouter-o4-mini" "openrouter-qwen3-next" )
# models=( "gemini-2.5-flash")
ROUNDS=5
MATRIX=0

cd ../

for MODEL in "${models[@]}"; do
    uv run src/p1_reliability_test.py \
        --model "$MODEL" \
        --n_repeats "$ROUNDS" \
        --out_path "data/p1_reliability_matrix_${MATRIX}/${MODEL}/" \
        --matrix_number "$MATRIX" \
        --payoff_matrix_path "src/test.json" \
        --verbose
    sleep 10
done
