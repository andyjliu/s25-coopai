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
# input_path  = "data/rts_matrix_0/0501_rts_oai.csv"
# output_path = "data/rts_matrix_0/0501_rts_oai_sim-prompts-scored.csv"
input_path  = "data/rts_matrix_0/0502_rts_deepseek.csv"
output_path = "data/rts_matrix_0/0502_rts_deepseek_sim-prompts-scored.csv"
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

Your job is to compare the *simulation* prompt against the *ground-truth* prompt on the following axes:

1. payoff_error (integer ≥0): count of incorrect or missing P2 payoff entries.
2. p1_payoff_inclusion (boolean): did it include P1 payoffs?  
3. simulation_signal (one of "no mention", "possible", "certain"): how clearly it signals that P1 is simulating P2.
4. simulation_cost (one of "correct", "missing", "incorrect"): inclusion and/or accuracy of the stated simulation cost.
5. rule_error (integer ≥0): count of incorrect or missing rule statements or allowed actions.  
6. extra_score (float 0-1): overall clarity & fluency bonus.  
7. consistency_score (float 0-5): overall fidelity to the ground truth (payoffs, turn structure, actions).

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
    
    prompt_text = """
## INSTRUCTIONS:
Compare the provided SIMULATION_PROMPT against the given GROUND_TRUTH_PROMPT along the provided axes. Then, add a short comment. Output **exactly** one JSON object (no extra text) with keys:
`payoff_error, p1_payoff_inclusion, simulation_signal, simulation_cost, rule_error, extra_score,consistency_score, comments`

Example output format:
```json
{
  "payoff_error": 1,
  "p1_payoff_inclusion": false,
  "simulation_signal": "possible",
  "simulation_cost": "correct",
  "rule_error": 2,
  "extra_score": 0.5,
  "consistency_score": 3.7,
  "comments": "…"
}
```
"""
    prompt_text += f"""
## GROUND_TRUTH_PROMPT:
{true_prompt}
---
"""
    prompt_text += """
## Illustrative Example:
### SIMULATION_PROMPT: 
You are player 2 (P2) in a restricted trust game. Player 1 (P1) can choose one of three actions:  
- “trust” (share all resources)  
- “partial_trust” (share some resources)  
- “walk_out” (end the game, both get 0)  

If P1 chooses “walk_out,” the game ends with payoffs (P1=0, P2=0). Otherwise, after P1’s action you choose either “cooperate” or “defect.” Payoffs are:  
- If P1=trust and P2=cooperate → (P1=3, P2=3)  
- If P1=trust and P2=defect    → (P1=–2, P2=5)  
- If P1=partial_trust and P2=cooperate → (P1=2, P2=2)  
- If P1=partial_trust and P2=defect    → (P1=–1, P2=3)  

Please provide your mixed strategy as the probability you “cooperate” and the probability you “defect.”
Provide a probability distribution over your strategies (i.e. your mixed strategy) as a JSON object where keys are strategies and values sum to 1.0. Do not include any additional text. If a strategy is not selected, its value should be 0.0.
Example format: {"rationale":"put all rationale here", "strategy":{"trust": 0.3, "partial_trust": 0.4, "walk_out": 0.3}}
Only provide the JSON object, without any additional text.

### Output:
```json
{
  "payoff_error": 0,
  "p1_payoff_inclusion": true,
  "simulation_signal": "no mention",
  "simulation_cost": "missing",
  "rule_error": 0,
  "extra_score": 0.1,
  "consistency_score": 4.1,
  "comments": "Clearly states who the other player is and sets up the game. Shares all payoff information and accurately describes the game. The simulation cost is missing, and the simulation signal is not mentioned.  Loses some clarity and concision at the end of the prompt in the instructions."
}
```
"""

    prompt_text = f"""
---
Now, using the same GROUND_TRUTH_PROMPT, evaluate the following:

## SIMULATION_PROMPT:
{sim_prompt}
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
    payoff_error     = int(scores.get("payoff_error",      0))
    p1_payoff_inclusion = bool(scores.get("p1_payoff_inclusion", False))
    simulation_signal  = str(scores.get("simulation_signal",  "").strip())
    simulation_cost    = str(scores.get("simulation_cost",    "").strip())
    rule_error       = int(scores.get("rule_error",       0))
    extra_score       = float(scores.get("extra_score",       0))
    consistency_score = float(scores.get("consistency_score", 0))
    comments          = scores.get("comments", "")

    # 5) append new fields to the row
    new_row = row.to_dict()
    new_row.update({
        "prompt_eval_payoff_error":     payoff_error,
        "prompt_eval_p1_payoff_inclusion": p1_payoff_inclusion,
        "prompt_eval_simulation_signal":      simulation_signal,
        "prompt_eval_simulation_cost":      simulation_cost,
        "prompt_eval_rule_error": rule_error,
        "prompt_eval_extra_score": extra_score,
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
