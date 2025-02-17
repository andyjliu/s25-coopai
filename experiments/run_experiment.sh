MODEL=gemini-1.5-flash
AGENT1=tit_for_tat
AGENT2=random

uv run src/prisoners_dilemma.py \
  --model "$MODEL" \
  --rounds 5 \
  --agent1 "$AGENT1" \
  --agent2 "$AGENT2" \
  --temperature 0.7 \
  --output "prisoners_dilemma_results_${MODEL}_${AGENT1}_${AGENT2}.csv" \
  --predict
