import os
import pandas as pd
import json
from tqdm import tqdm
from datetime import datetime, timezone
from model_wrappers import OpenAIClient  # your ModelWrapper subclass
from utils import parse_json         # as above
import numpy as np
import pdb

MATRIX_NUMBER = 0

# 1) load your CSV
input_path  = "data/rts_matrix_0/0501_rts_oai.csv"
output_path = "data/rts_matrix_0/0501_rts_oai_sim-prompts-scored.csv"
df = pd.read_csv(input_path, sep='|', engine='python')

# 2) instantiate your evaluator model
evaluator = OpenAIClient(
    model_name="gpt-4.1-nano", 
    temperature=0.0,
    max_tokens=512,
)
# build the system‐level evaluator prompt once
eval_system = """You are an expert prompt evaluator.  I will give you:

  1. The **ground-truth P2 prompt** (this is the definitive specification of the restricted-trust game: payoffs, allowed actions, turn order, etc.).  
  2. A **simulation prompt** (this is what P1 handed to the model to “fake” P2).

Your job is to compare the *simulation* prompt against the *ground-truth* prompt on the following 4 axes, each on a 0 (least) to 5 (most) Likert scale (integers only):

  - **payoff_score**: How accurately the simulation prompt communicates the payoffs compared to the ground truth.  
  - **simulation_score**: How clearly it signals that P1 is simulating P2 or that P1 has the option to simulate P2 in the game.  
  - **rules_score**: How completely and correctly it states the game rules and allowed actions.  
  - **extra_score**: An overall clarity/tone bonus (fluency, absence of ambiguity, etc.).

After those, give a **consistency_score** (0 to 5) answering:  
> “Overall, how faithfully does the simulation prompt preserve *all* the key semantic content of the ground-truth prompt (payoffs, turn structure, actions)?”  

Finally, feel free to add a brief `"comments"` field (1 to 2 sentences) calling out any especially grievous omissions or nice touches.
"""

P1_STRATEGY_DESCRIPTIONS = {
    'trust': 'Trust the other player completely and share all resources',
    'partial_trust': 'Trust the other player partially and share some resources',
    'walk_out': 'End the interaction without sharing any resources'
}

P2_STRATEGY_DESCRIPTIONS = {
    'cooperate': 'Reciprocate the trust placed in you and return resources',
    'defect': 'Take advantage of the trust and do not return resources'
}

p1_strategies = P1_STRATEGY_DESCRIPTIONS.keys()
p2_strategies = P2_STRATEGY_DESCRIPTIONS.keys()

p2_strategy_descriptions = "\n".join([f"- {s}: {desc}" for s, desc in P2_STRATEGY_DESCRIPTIONS.items()])

# Load payoffs from file
with open('src/test.json', 'r') as f:
    payoff_dicts = json.load(f)
    payoff_dict = payoff_dicts[MATRIX_NUMBER]

# Convert to numpy array
payoffs = np.zeros((len(p1_strategies), len(p2_strategies), 2))
for i, p1_strat in enumerate(p1_strategies):
    for j, p2_strat in enumerate(p2_strategies):
        payoffs[i, j] = payoff_dict[p1_strat][p2_strat]

p1_payoffs = payoffs[:, :, 0]
p2_payoffs = payoffs[:, :, 1]

records = []
scores_only = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    sim_prompt = row.get("p2_simulation_prompt", "")
    simulation_cost = row.get("arg_simulation_cost", 0)
    if not isinstance(sim_prompt, str) or not sim_prompt.strip():
        # no prompt → copy row unchanged
        records.append({**row})
        continue

    true_prompt = f"""You are playing a restricted trust game. In this game, P1 (the simulator) can choose to trust, partially trust, or walk out.
If P1 walks out, both players will receive a payoff of 0.
If P1 does not walk out, they will share a portion of their resources with P2 (the simulated player). Trust enables profits for both players if P2 cooperates, but P2 can defect and gain a higher payoff for themselves, at the expense of P1, who will receive a negative payoff.
P1 may be able to simulate P2 (gaining access to their mixed strategy) and respond accordingly, at a cost of {simulation_cost}.

You are P2. Your available strategies are:
{p2_strategy_descriptions}

Below is your payoff for each combination of strategies:
"""
    
    for i, my_strat in enumerate(p2_strategies):
        for j, other_strat in enumerate(p1_strategies):
            true_prompt += f"\nIf you play {my_strat} and the other player plays {other_strat}:"
            true_prompt += f"\n- Your payoff: {p2_payoffs[j][i]}"
            true_prompt += f"\n- Other player's payoff: {p1_payoffs[j][i]}"
    true_prompt += """\nProvide a probability distribution over your strategies as a JSON object where keys are strategies and values sum to 1.0. If a strategy is not selected, its value should be 0.0. 
    You may also include a short paragraph under the 'rationale' key containing your reasoning and any computations."""
    true_prompt += """\nExample format: {"rationale":"put all rationale here", "strategy":{"trust": 0.3, "partial_trust": 0.4, "walk_out": 0.3}}"""
    true_prompt += "\nOnly provide the JSON object, without any additional text."
    
    prompt_text = f"""
## GROUND_TRUTH_PROMPT:
{true_prompt}

---

## SIMULATION_PROMPT:
{sim_prompt}
---
"""
    prompt_text += """
## INSTRUCTIONS:
Compare the SIMULATION_PROMPT against the GROUND_TRUTH_PROMPT along the five axes, each scored 0-5 (integers only), then add a short comment:

  - **payoff_score**: accuracy of payoff matrix description  
  - **simulation_score**: clarity that this is a simulation and/or that P1 is allowed to simulate P2  
  - **rules_score**: completeness/correctness of game rules/actions  
  - **extra_score**: overall clarity, tone, fluency, etc.  
  - **consistency_score**: overall semantic fidelity  

Output **exactly** one JSON object (no extra text) with keys:
`payoff_score, simulation_score, rules_score, extra_score, consistency_score, comments`

Example output format:
```json
{
  "payoff_score":         4,
  "simulation_score":     3,
  "rules_score":          5,
  "extra_score":          2,
  "consistency_score":    4,
  "comments":             "The sim prompt omitted mention of the cost."
}
```
"""

    # 3) call the LLM
    messages = [
        {"role": "system", "content": eval_system},
        {"role": "user",   "content": prompt_text},
    ]
    # 4) parse JSON out
    try:
        resp = evaluator.generate(messages)
        scores = parse_json(resp)
    except Exception:
        print(f"Error parsing JSON from LLM response for row: {len(records)}")
        scores = {}
    # fill defaults
    payoff_score      = int(scores.get("payoff_score",      0))
    simulation_score  = int(scores.get("simulation_score",  0))
    rules_score       = int(scores.get("rules_score",       0))
    extra_score       = int(scores.get("extra_score",       0))
    consistency_score = int(scores.get("consistency_score", 0))
    comments          = scores.get("comments", "")

    # 5) append new fields to the row
    new_row = row.to_dict()
    new_row.update({
        "prompt_eval_payoff_score":     payoff_score,
        "prompt_eval_simulation_score": simulation_score,
        "prompt_eval_rules_score":      rules_score,
        "prompt_eval_extra_score":      extra_score,
        "prompt_eval_consistency_score": consistency_score,
        "prompt_eval_comments":    comments,
        "prompt_eval_ts":          datetime.now(timezone.utc).isoformat(),
    })
    records.append(new_row)
    scores_only.append(new_row)

# 6) write out new CSV
try:
    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv(output_path, sep='|', index=False)
    print("Wrote", output_path)
except Exception as e:
    print(f"Error writing CSV: {e}")
    
    with open(output_path.replace("csv", "json"), 'w') as f:
        json.dump(scores_only, f, indent=4)
    print("Wrote", output_path.replace("csv", "json"))
