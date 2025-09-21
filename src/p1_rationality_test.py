# file: logical_response_experiment.py
from __future__ import annotations

import json, csv, pdb, os
from dataclasses import dataclass
from utils import parse_json, solve_stackelberg_game, validate_trust_game
from model_wrappers import ModelWrapper, Message
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
from restricted_trust_with_simulation import SimulatorAgent, P1_STRATEGY_DESCRIPTIONS, P2_STRATEGY_DESCRIPTIONS

# ---- Utilities ---------------------------------------------------------------

def l1_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    """L1 distance between two strategy dicts with same keys."""
    keys = list(p.keys())
    return float(sum(abs(p[k] - q[k]) for k in keys))

def normalize_probs(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, float(v)) for v in d.values())
    if s <= 0:
        # fallback to uniform
        n = len(d)
        return {k: 1.0 / n for k in d}
    return {k: max(0.0, float(v)) / s for k, v in d.items()}

# ---- Experiment --------------------------------------------------------------

@dataclass
class SimulationScenario:
    """A pre-set simulated result for P2 (diverse mixes)."""
    label: str
    strategy: Dict[str, float]  # over P2 strategies (e.g., {'cooperate': 0.8, 'defect': 0.2})

class LogicalResponseExperiment:
    """
    Experiment: P1 decides whether to simulate. If P1 chooses to simulate, we feed a pre-set
    'simulation result' (a P2 mixed strategy), then re-prompt P1 for a response *without*
    auto-best-responding. We compare P1's response to the true best response, log deviations.
    """

    def __init__(
        self,
        p1_agent,                         # SimulatorAgent (or compatible)
        p1_strategies: List[str],         # e.g., ['trust','partial_trust','walk_out']
        p2_strategies: List[str],         # e.g., ['cooperate','defect']
        p1_payoffs: np.ndarray,           # shape (len(p1_strategies), len(p2_strategies))
        p2_payoffs: np.ndarray,           # shape (len(p1_strategies), len(p2_strategies))  (not used here but handy)
        simulation_cost: float,
        scenarios: Optional[List[SimulationScenario]] = None,
        seed: Optional[int] = None,
    ):
        self.p1 = p1_agent
        self.p1_strategies = p1_strategies
        self.p2_strategies = p2_strategies
        self.p1_payoffs = p1_payoffs
        self.p2_payoffs = p2_payoffs
        self.simulation_cost = simulation_cost

        if seed is not None:
            np.random.seed(seed)

        # Diverse defaults if none provided
        if scenarios is None:
            scenarios = [
                SimulationScenario("uniform",       {"cooperate": 0.5, "defect": 0.5}),
                SimulationScenario("coop_heavy",    {"cooperate": 0.8, "defect": 0.2}),
                SimulationScenario("defect_heavy",  {"cooperate": 0.2, "defect": 0.8}),
                SimulationScenario("near_coop_pure",{"cooperate": 0.95,"defect": 0.05}),
                SimulationScenario("near_def_pure", {"cooperate": 0.05,"defect": 0.95}),
            ]
        # Normalize to be safe
        self.scenarios = [
            SimulationScenario(s.label, normalize_probs(s.strategy)) for s in scenarios
        ]

    # ---------- Prompt builders ----------------------------------------------

    def build_response_prompt(
        self,
        initial_prompt: str,
        chosen_sim_result: SimulationScenario,
    ) -> str:
        """
        Build the second prompt that asks P1 to respond *after* seeing the simulated P2 strategy.
        P1 returns JSON with 'rationale' and 'strategy' over P1 strategies ONLY (no 'simulate' key).
        """
        history = (
            "You previously received this prompt (for your initial decision):\n"
            "```" + initial_prompt + "```\n\n"
        )

        sim_context = (
            "You chose to simulate, so you now receive the other player's mixed strategy "
            f"(from the simulation): {chosen_sim_result.strategy}.\n\n"
        )

        instructions = (
            "Given the above, please choose **your** response *now*.\n"
            "Important:\n"
            "  - You must NOT include any 'simulate' action this time.\n"
            "  - Return a JSON object with keys:\n"
            "      'rationale' : short reasoning including any computation\n"
            "      'strategy'  : your mixed strategy over P1 strategies ONLY.\n"
            f"    P1 strategies are: {self.p1_strategies}\n"
            "  - Probabilities must be nonnegative and sum to 1.0.\n"
            'Example:\n'
            '{"rationale":"short reasoning and math here", '
            '"strategy":{"trust":0.6,"partial_trust":0.4,"walk_out":0.0}}\n'
            "Only provide the JSON object, with no extra text."
        )

        return history + sim_context + instructions

    # ---------- Core runner ---------------------------------------------------

    def run(
        self,
        n_repeats_per_scenario: int,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        For each scenario, run n repeats:
          1) P1 is prompted normally (which can include 'simulate')
          2) If P1 did NOT choose 'simulate', we record a 'no-simulation' trial (baseline)
          3) If P1 DID choose 'simulate', we feed the pre-set scenario's P2 strategy
             and re-prompt P1 for a response (no 'simulate' allowed here)
          4) Compute the best response and deviations (L1, TV), plus expected-payoff shortfall
          5) Log everything to CSV and return a DataFrame
        """
        rows = []

        for scenario in self.scenarios:
            for trial in range(n_repeats_per_scenario):
                # ---- Step 1: initial prompt as usual
                initial_prompt = self.p1.get_strategy_elicitation_prompt(default_format=True)

                # ---- Step 2: (P1 did choose to simulate) provide scenario strategy and re-prompt
                response_prompt = self.build_response_prompt(
                    initial_prompt=initial_prompt,
                    chosen_sim_result=scenario,
                )

                try:
                    response_json = parse_json(self.p1.model.generate([{"role": "user", "content": response_prompt}]))
                    if verbose:
                        print(f"Model response JSON: {response_json}")
                except Exception as e:
                    # Fallback: walk_out
                    if verbose:
                        print(f"Error parsing JSON response from model: {e}. Falling back to 'walk_out'.")
                    
                    response_json = {
                        "rationale": f"ERROR: {e}",
                        "strategy": {s: 0.0 for s in self.p1_strategies}
                    }
                    response_json["strategy"][self.p1_strategies[-1]] = 1.0

                resp_strategy = normalize_probs(response_json.get("strategy", {}))
                resp_rationale = response_json.get("rationale", "")

                # ---- Step 3: compute best response to the scenario strategy
                # Convert scenario strategy into array in P2-strategy order:
                p2_probs = np.array([scenario.strategy[k] for k in self.p2_strategies], dtype=float)

                # Expected payoff for each P1 pure strategy
                exp_payoffs = np.dot(self.p1_payoffs, p2_probs)
                max_payoff = float(np.max(exp_payoffs))

                # best-response mixed (equal split over any ties)
                best_mask = np.isclose(exp_payoffs, max_payoff, rtol=1e-10)
                best_probs = best_mask.astype(float) / np.sum(best_mask)
                best_response = dict(zip(self.p1_strategies, best_probs.tolist()))

                # Deviations
                deviation_l1 = l1_distance(resp_strategy, best_response)
                tv_distance = 0.5 * deviation_l1

                # Expected payoff under P1's **response** vs **best response**
                resp_prob_vec = np.array([resp_strategy[k] for k in self.p1_strategies], dtype=float)
                best_prob_vec = np.array([best_response[k] for k in self.p1_strategies], dtype=float)

                exp_payoff_resp = float(resp_prob_vec @ exp_payoffs)
                exp_payoff_best = float(best_prob_vec @ exp_payoffs)
                shortfall = float(exp_payoff_best - exp_payoff_resp)

                # ---- Log
                rows.append({
                    "scenario_label": scenario.label,
                    "scenario_p2_strategy": json.dumps(scenario.strategy),
                    "trial": trial,
                    "response_trust": resp_strategy.get("trust", 0.0),
                    "response_partial_trust": resp_strategy.get("partial_trust", 0.0),
                    "response_walk_out": resp_strategy.get("walk_out", 0.0),
                    "response_rationale": resp_rationale,
                    "response_strategy": json.dumps(resp_strategy),
                    "best_response_trust": best_response.get("trust", 0.0),
                    "best_response_partial_trust": best_response.get("partial_trust", 0.0),
                    "best_response_walk_out": best_response.get("walk_out", 0.0),
                    "best_response": json.dumps(best_response),
                    "l1_deviation": deviation_l1,
                    "tv_distance": tv_distance,
                    "exp_payoff_response": exp_payoff_resp,
                    "exp_payoff_best": exp_payoff_best,
                    "exp_payoff_shortfall": shortfall,
                })

        df = pd.DataFrame(rows)
        return df, rows

def write_summary_to_csv(results, csv_path: str, args=None):
        """
        Write each individual game round to a CSV file as separate rows
        If the file doesn't exist, create it with headers
        If it exists, append the new rows
        
        Args:
            csv_path: Path to CSV file
            args: Optional argparse namespace with experiment parameters
        """
        
        # Check if file exists to determine if headers need to be written
        file_exists = os.path.isfile(csv_path)
        
        # Define all possible column names
        columns = [
            # Round data
            "timestamp",
            "total_index",
            "scenario_label",
            "scenario_p2_strategy",
            "trial",
            "response_trust",
            "response_partial_trust",
            "response_walk_out",
            "response_strategy",
            "response_rationale",
            "best_response_trust",
            "best_response_partial_trust",
            "best_response_walk_out",
            "best_response",
            "l1_deviation",
            "tv_distance",
            "exp_payoff_response",
            "exp_payoff_best",
            "exp_payoff_shortfall",
            # Command line args (if any)
        ]
        
        # Add command line arguments if provided
        if args:
            for arg in vars(args):
                columns.append(f"arg_{arg}")
        
        # Prepare data for writing
        rows = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for round_idx, round_data in enumerate(results):
            row = {
                "timestamp": timestamp,
                "total_index": round_idx,
                "scenario_label": round_data["scenario_label"],
                "scenario_p2_strategy": round_data["scenario_p2_strategy"],
                "trial": round_data["trial"],
                "response_trust": round_data["response_trust"] if isinstance(round_data["response_trust"], float) else None,
                "response_partial_trust": round_data["response_partial_trust"] if isinstance(round_data["response_partial_trust"], float) else None,
                "response_walk_out": round_data["response_walk_out"] if isinstance(round_data["response_walk_out"], float) else None,
                "response_strategy": round_data["response_strategy"],
                "response_rationale": round_data["response_rationale"],
                "best_response_trust": round_data["best_response_trust"] if isinstance(round_data["best_response_trust"], float) else None,
                "best_response_partial_trust": round_data["best_response_partial_trust"] if isinstance(round_data["best_response_partial_trust"], float) else None,
                "best_response_walk_out": round_data["best_response_walk_out"] if isinstance(round_data["best_response_walk_out"], float) else None,
                "best_response": round_data["best_response"],
                "l1_deviation": round_data["l1_deviation"] if isinstance(round_data["l1_deviation"], float) else None,
                "tv_distance": round_data["tv_distance"] if isinstance(round_data["tv_distance"], float) else None,
                "exp_payoff_response": round_data["exp_payoff_response"] if isinstance(round_data["exp_payoff_response"], float) else None,
                "exp_payoff_best": round_data["exp_payoff_best"] if isinstance(round_data["exp_payoff_best"], float) else None,
                "exp_payoff_shortfall": round_data["exp_payoff_shortfall"] if isinstance(round_data["exp_payoff_shortfall"], float) else None,
            }
                
            # Add command line arguments if provided
            if args:
                for arg, value in vars(args).items():
                    row[f"arg_{arg}"] = value
                    
            rows.append(row)
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        # Write to CSV
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns, delimiter='|')
            
            # Write headers only if file is new
            if not file_exists:
                writer.writeheader()
                
            # Write all rows
            writer.writerows(rows)

if __name__ == "__main__":
    """
    This block shows how you might wire the experiment up with your existing agents.
    Assumes you have:
      - a SimulatorAgent-like P1 with .model, .get_strategy_elicitation_prompt(), .payoffs, etc.
      - payoff matrices p1_payoffs, p2_payoffs
      - strategy lists p1_strategies, p2_strategies
    Replace placeholders with your actual objects.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Model name for ModelWrapper')
    parser.add_argument('--payoff_matrix_path', type=str, required=True, help='Path to JSON file with payoff matrices')
    parser.add_argument('--matrix_number', type=int, required=True, help='Key of the payoff matrix to use')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to output CSV file')
    parser.add_argument('--n_repeats', type=int, default=1, help='Number of repeats for each scenario')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for the model')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    model = ModelWrapper.create(args.model_name, temperature=args.temperature)
    with open(args.payoff_matrix_path, 'r') as f:
        payoff_dicts = json.load(f)
        payoff_dict = payoff_dicts[args.matrix_number]
        assert validate_trust_game(payoff_dict), "Invalid payoff matrix."
    
    p1_strategies = list(P1_STRATEGY_DESCRIPTIONS.keys())
    p2_strategies = list(P2_STRATEGY_DESCRIPTIONS.keys())
    payoffs = np.zeros((len(p1_strategies), len(p2_strategies), 2))
    for i, p1_strat in enumerate(p1_strategies):
        for j, p2_strat in enumerate(p2_strategies):
            payoffs[i, j] = payoff_dict[p1_strat][p2_strategies[j]]
    p1_payoffs = payoffs[:, :, 0]
    p2_payoffs = payoffs[:, :, 1]

    p1 = SimulatorAgent(
        name="P1",
        model=model,
        strategies=list(P1_STRATEGY_DESCRIPTIONS.keys()),
        p1_payoffs=p1_payoffs,
        p2_payoffs=p2_payoffs,
        simulation_cost=0.0,
        simulation_type="simulate_internally",  # not used here, but fine
    )
    
    exp = LogicalResponseExperiment(
        p1_agent=p1,
        p1_strategies=list(P1_STRATEGY_DESCRIPTIONS.keys()),
        p2_strategies=list(P2_STRATEGY_DESCRIPTIONS.keys()),
        p1_payoffs=p1_payoffs,
        p2_payoffs=p2_payoffs,
        simulation_cost=0.0,
        seed=42,
    )

    df, rows = exp.run(n_repeats_per_scenario=args.n_repeats, verbose=args.verbose)
    write_summary_to_csv(rows, args.csv_path, args)
    
    if args.verbose:
        print(f"Results written to {args.csv_path}")
        print(df.head())
        print(f"Average regret (shortfall): {df['exp_payoff_shortfall'].mean()}")
        print(f"Average TV distance: {df['tv_distance'].mean()}")