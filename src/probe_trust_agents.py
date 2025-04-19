#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import pdb

from utils import parse_json
from model_wrappers import ModelWrapper
from restricted_trust_with_simulation import (
    SimulatorAgent, SimulatedAgent,
    P1_STRATEGY_DESCRIPTIONS, P2_STRATEGY_DESCRIPTIONS
)

def load_payoff_matrix(path, p1_strats, p2_strats):
    """Load payoffs from JSON into separate P1 and P2 numpy matrices."""
    with open(path) as f:
        data = json.load(f)
    m, n = len(p1_strats), len(p2_strats)
    p1pm = np.zeros((m, n))
    p2pm = np.zeros((m, n))
    for i, s1 in enumerate(p1_strats):
        for j, s2 in enumerate(p2_strats):
            pay = data[s1][s2]
            p1pm[i, j] = pay[0]
            p2pm[i, j] = pay[1]
    return p1pm, p2pm

def build_p1_probe_prompt(
        agent: SimulatorAgent, 
        other_strategy: dict, 
        question: str
    ) -> str:
    """
    Start from the agent’s own elicitation prompt, then append:
      1) The other player's fixed mixed strategy
      2) A request for JSON + natural‐language explanation
    """
    # 1) Base: what the agent already knows how to describe
    prompt = agent.get_strategy_elicitation_prompt(default_format = False)
    
    if other_strategy is None:
        prompt += f"""\nProvide
    (1) a brief natural-language explanation of your reasoning
    (2) a probability distribution over your strategies and 'simulate' as a JSON object where keys are strategies and values sum to 1.0
    {'(3) your answer to the question:' + question if question else ''}\n"""
        if question:
            prompt += """Example format: {"rationale": "fill this in rationale and any computations", "strategy": {"strategy1": 0.5, "strategy2": 0.3, "simulate": 0.2}, "question": "fill this in with your answer"}"""
        else:
            prompt += """Example format: {"rationale": "fill this in rationale and any computations", "strategy": {"strategy1": 0.5, "strategy2": 0.3, "simulate": 0.2}}"""
    else:
        # 2) Add the fixed‐opponent context in the case where we have it
        prompt = "Earlier you were provided with the following information about the game you are playing: \n" + f"'{prompt}'\n"
        prompt += "\n\nYou chose to simulate the other player and in simulation, other player's mixed strategy was: ["
        strategy_list = []
        for strat, prob in other_strategy.items():
            strategy_list.append(f"{strat}: {prob:.2f}")
        prompt += ", ".join(strategy_list) + "]\n"
        
        # 3) Close with exactly what we want back
        prompt += f"""\nPlease provide:\n
    (1) The computation and reasoning you did to determine your best response to the other player’s strategy, accounting for any potential truthfulness or deception of other player in the simulation.
    (2) A JSON object giving your probability distribution over your available strategies
    {'(3) Answer the question: '+question if question else ''}\n"""
        if question:
            prompt +="""Example format: {"rationale": "fill this in with computation and rationale", "strategy": {"strategy1": 0.5, "strategy2": 0.3}, "question": "fill this in with your answer"}"""
        else:
            prompt +="""Example format: {"rationale": "fill this in with computation and rationale", "strategy": {"strategy1": 0.5, "strategy2": 0.3}}"""

    print(f"Prompt for P1: {prompt}")
    return prompt

def run_p1_probe(agent, contexts, additional_question=None):
    """
    Given an agent and a list of `contexts` (dict of other_strategy),
    send each prompt to the LLM, parse its JSON response and collect rationale.
    """
    results = []
    for ctx in contexts:
        prompt = build_p1_probe_prompt(agent, ctx, additional_question)
        resp = agent.model.generate([{"role":"user","content": prompt}])
        text = resp.strip()
        try:
            parsed = parse_json(text)
            if "strategy" in parsed and isinstance(parsed["strategy"], dict):
                dist = parsed["strategy"]
            else:
                dist = {s: 1.0 / len(agent.strategies) for s in agent.strategies}
            rationale = parsed.get("rationale", "")
            answer    = parsed.get("question", None)
        except Exception:
            # fallback to uniform
            dist = {s:1/len(agent.strategies) for s in agent.strategies}
            rationale = f"Failed to parse JSON: {text}"
            answer = None

        entry = {
            "agent":           agent.name,
            "agent_type":      agent.__class__.__name__,
            "p2_strategy":     ctx,
            "distribution":    dist,
            "rationale":       rationale,
        }

        if additional_question:
            entry["question"] = additional_question
            entry["answer"]   = answer
        
        results.append(entry)
    return results

def build_p2_probe_prompt(
    agent: SimulatedAgent,
    p1_strategy: dict = None,
    in_simulation: str = "Unknown",
    p1_sim_trust: float = None,
    question: str = None
) -> str:
    """
    Build a probe prompt for Player 2.  Uses agent.get_strategy_elicitation_prompt()
    as a base, then optionally adds:
      - a fixed P1 mixed strategy (if p1_strategy is provided)
      - whether we're in simulation and how much P1 trusts it
      - an optional follow‑up question

    Returns the full prompt string.
    """
    # 1) start with the agent’s native prompt (no automatic JSON/format block)
    #    we assume get_strategy_elicitation_prompt accepts default_format=False
    base = agent.get_strategy_elicitation_prompt(default_format=False)

    prompt = base + "\n\n"

    # 2) If P1’s strategy is known:
    if p1_strategy is not None:
        prompt += "You know that P1 is playing the following fixed mixed strategy:\n"
        for strat, prob in p1_strategy.items():
            prompt += f"- {strat}: {prob:.2f}\n"

    # 3) Simulation context
    if in_simulation == "Known":
        prompt += f"\nYou are currently in a simulation.{' P1 trusts the simulation with probability '+ p1_sim_trust:.2f if p1_sim_trust is not None else ''}.\n"
        if p1_sim_trust is not None:
            prompt += f""
    elif in_simulation == "Possible":
        prompt += f"P1 is able to simulate you{' and would trust the simulation result with probability ' + p1_sim_trust:.2f if p1_sim_trust is not None else ''}. You may or may not be in simulation currently.\n"        
    else:
        pass

    # 4) What we want back
    prompt += "\nPlease provide as a single JSON dictionary:\n"
    prompt += "  (1) A description of your reasoning and any computations (under the \"rationale\" key).\n"
    prompt += "  (2) A dictionary under key \"strategy\" giving your probability distribution over your strategies."
    if question:
        prompt += f"\n  (3) Your answer to the question: {question}\n"
    else:
        prompt += "\n"
    # Example format
    if question:
        prompt += (
            'Example format:\n'
            '{"rationale":"put all rationale and computation here","strategy":{"cooperate":0.6,"defect":0.4},"question":"..."}\n'
        )
    else:
        prompt += (
            'Example format:\n'
            '{"rationale":"put all rationale and computation here","strategy":{"cooperate":0.6,"defect":0.4}}\n'
        )

    print(f"Prompt for P2: {prompt}")
    return prompt

def run_p2_probe(
    agent: SimulatedAgent,
    contexts: list,
    in_simulation: bool = False,
    sim_trust: float = None,
    question: str = None
) -> list:
    """
    Probe P2 under each P1-strategy context.  `contexts` is a list of dicts
    mapping P1 strategies to probabilities (or [None] to omit).
    Returns a list of result dicts with keys:
      agent, agent_type, p1_strategy, distribution, rationale, [question, answer]
    """
    results = []
    for ctx in contexts:
        prompt = build_p2_probe_prompt(
            agent,
            p1_strategy=ctx,
            in_simulation=in_simulation,
            p1_sim_trust=sim_trust,
            question=question
        )

        resp = agent.model.generate([{"role":"user","content": prompt}])
        text = resp.strip()

        # Extract JSON blob with parse_json
        parsed = parse_json(text)

        # Strategy distribution
        strat_dist = parsed.get("strategy")
        if not isinstance(strat_dist, dict):
            # fallback to uniform
            strat_dist = {s: 1/len(agent.strategies) for s in agent.strategies}

        # Rationale
        rationale = parsed.get("rationale", "")

        # Optional question answer
        ans = parsed.get("question", None)

        entry = {
            "agent":        agent.name,
            "agent_type":   agent.__class__.__name__,
            "p1_strategy":  ctx,
            "distribution": strat_dist,
            "rationale":    rationale
        }
        if question:
            entry["question"] = question
            entry["answer"]   = ans

        results.append(entry)

    return results

def probe_p1(args):
    """Probe Player 1 by feeding it fixed P2 strategies."""
    p1_strats = list(P1_STRATEGY_DESCRIPTIONS.keys())
    p2_strats = list(P2_STRATEGY_DESCRIPTIONS.keys())
    p1_payoffs, _ = load_payoff_matrix(args.payoff_matrix, p1_strats, p2_strats)

    p1_model = ModelWrapper.create(args.p1_model, temperature=args.temperature)
    p1_agent = SimulatorAgent(
        "P1", p1_model, p1_strats, p1_payoffs,
        simulation_cost=args.simulation_cost,
        simulation_type=args.simulation_type
    )

    # contexts: each pure P2 strat + uniform
    contexts = [None]+[
        {s: 1.0 if s==ps else 0.0 for s in p2_strats}
        for ps in p2_strats
    ] + [{s: 1/len(p2_strats) for s in p2_strats}]

    return run_p1_probe(p1_agent, contexts)

def probe_p2(args):
    """Probe Player 2 by feeding it fixed P1 strategies."""
    p1_strats = list(P1_STRATEGY_DESCRIPTIONS.keys())
    p2_strats = list(P2_STRATEGY_DESCRIPTIONS.keys())
    _, p2_payoffs = load_payoff_matrix(args.payoff_matrix, p1_strats, p2_strats)

    p2_model = ModelWrapper.create(args.p2_model, temperature=args.temperature)
    p2_agent = SimulatedAgent("P2", p2_model, p2_strats, p2_payoffs)

    # contexts: each pure P1 strat + uniform
    contexts = [None]+[
        {s: 1.0 if s==ps else 0.0 for s in p1_strats}
        for ps in p1_strats
    ] + [{s: 1/len(p1_strats) for s in p1_strats}]

    results = []
    sim_trusts = [0.0, 0.5, 1.0]
    is_sim = ["Unknown", "Possible", "Known"]
    for trust in sim_trusts:
        for sim in is_sim:
            res = run_p2_probe(p2_agent, contexts)
            results.append({"is_simulation": sim, 
                            "sim_trust": trust, 
                            "results": res})
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["p1","p2"], required=True,
                        help="Which agent to probe")
    parser.add_argument("--p1_model", default="gpt-4o-mini")
    parser.add_argument("--p2_model", default="gpt-4o-mini")
    parser.add_argument("--payoff_matrix", default="src/test.json",
                        help="JSON file with payoff dict")
    parser.add_argument("--simulation_cost", type=float, default=0.5)
    parser.add_argument("--simulation_type", default="simulate_and_best_response")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output_dir", default="agent_probes/",
                        help="Where to write the probe results")
    args = parser.parse_args()

    print(f"Probing {args.agent} with model {args.p1_model if args.agent == 'p1' else args.p2_model}")
    if args.agent == "p1":
        results = probe_p1(args)
    else:
        results = probe_p2(args)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = args.output_dir + f"{args.agent}_probe.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} probe results to {output_file}")

if __name__ == "__main__":
    main()
