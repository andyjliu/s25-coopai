# MODEL=gemini-1.5-flash
# AGENT1=defensive
# AGENT2=cooperative

# uv run src/prisoners_dilemma.py \
#   --model "$MODEL" \
#   --rounds 5 \
#   --agent1 "$AGENT1" \
#   --agent2 "$AGENT2" \
#   --temperature 0.7 \
#   --output "prisoners_dilemma_results_${MODEL}_${AGENT1}_${AGENT2}.csv" \
#   --predict


# uv run src/explicit_simulation.py \
#   --model "$MODEL" \
#   --rounds 5 \
#   --agent1 "$AGENT1" \
#   --agent2 "$AGENT2" \
#   --temperature 0.7 \
#   --output "explicit_simulation_results_${MODEL}_${AGENT1}_${AGENT2}.csv" \
  # --simreasoning

sim_types=("simulate_externally" "simulate_and_best_response" "simulate_internally" "simulate_via_probing")
MODEL=gpt-4o-mini
ROUNDS=20
SIM_COST=1

cd ../
for SIM_TYPE in "${sim_types[@]}"; do
  echo "Running simulation type: $SIM_TYPE"
  uv run src/restricted_trust_with_simulation.py \
    --p1_model "$MODEL" \
    --p2_model "$MODEL" \
    --rounds $ROUNDS \
    --simulation_cost $SIM_COST \
    --simulation_type $SIM_TYPE \
    --csv_output "experiments/restricted_trust_results_models-${MODEL}_sim-type-${SIM_TYPE}_sim-cost-${SIM_COST}.csv" \
    --verbose
done