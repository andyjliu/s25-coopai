MODEL=gemini-1.5-flash
AGENT1=defensive
AGENT2=defensive

# uv run src/prisoners_dilemma.py \
#   --model "$MODEL" \
#   --rounds 5 \
#   --agent1 "$AGENT1" \
#   --agent2 "$AGENT2" \
#   --temperature 0.7 \
#   --output "prisoners_dilemma_results_${MODEL}_${AGENT1}_${AGENT2}.csv" \
#   --predict


uv run src/explicit_simulation.py \
  --model "$MODEL" \
  --rounds 5 \
  --agent1 "$AGENT1" \
  --agent2 "$AGENT2" \
  --temperature 0.7 \
  --output "explicit_simulation_results_${MODEL}_${AGENT1}_${AGENT2}.csv" \
  # --simreasoning