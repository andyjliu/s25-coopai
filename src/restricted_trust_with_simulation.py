import json
from typing import List, Dict, Tuple, Optional
from utils import parse_json
from model_wrappers import ModelWrapper, Message
from tqdm import tqdm

import pandas as pd
import numpy as np
import random
import cvxpy as cp

import os
import argparse
import pdb
import csv
from datetime import datetime


P1_STRATEGY_DESCRIPTIONS = {
    'trust': 'Trust the other player completely and share all resources',
    'partial_trust': 'Trust the other player partially and share some resources',
    'walk_out': 'End the interaction without sharing any resources'
}

P2_STRATEGY_DESCRIPTIONS = {
    'cooperate': 'Reciprocate the trust placed in you and return resources',
    'defect': 'Take advantage of the trust and do not return resources'
}

class RestrictedTrustAgent:
    def __init__(self, name: str, model: ModelWrapper, strategies: List[str], payoffs: np.ndarray):
        self.name = name
        self.model = model
        self.strategies = strategies
        self.payoffs = payoffs
        self.strategy_descriptions = P1_STRATEGY_DESCRIPTIONS if isinstance(self, SimulatorAgent) else P2_STRATEGY_DESCRIPTIONS

    def get_strategy_elicitation_prompt(self, default_format=True) -> str:
        raise NotImplementedError("Subclasses should implement this method")

    def get_mixed_strategy(self) -> Dict[str, float]:
        strategy_elicitation_prompt = self.get_strategy_elicitation_prompt()
        response = parse_json(self.model.generate([{"role": "user", "content": strategy_elicitation_prompt}]))
        strategy_list = self.strategies + ['simulate'] if isinstance(self, SimulatorAgent) else self.strategies
        try:
            assert len(response.keys()) > 0 and\
                abs(1 - sum([float(response.get(k, 0.0)) for k in response.keys()])) < 0.01, "Invalid distribution."
            return(
                {strategy: response[strategy] for strategy in strategy_list}
            )
        except (AssertionError, KeyError) as e:
            print(f"Error {e} processing model response: {response}. Returning uniform distribution")
            return(
                {strategy: 1.0/len(strategy_list) for strategy in strategy_list}
            )

class SimulatorAgent(RestrictedTrustAgent):
    def __init__(
        self, 
        name: str, 
        model: ModelWrapper, 
        strategies: List[str], 
        payoffs: np.ndarray,
        simulation_cost: float, 
        simulation_type: str
        ):
        super().__init__(name, model, strategies, payoffs)
        self.simulation_cost = simulation_cost
        self.simulation_type = simulation_type

    def get_strategy_elicitation_prompt(self, default_format=True) -> str:
        """Generate prompt for model to elicit strategy distribution"""
        strategy_descriptions = "\n".join([f"- {s}: {desc}" for s, desc in self.strategy_descriptions.items()])
        
        prompt = f"""You are playing a restricted trust game as {self.name}. Your available strategies are:
{strategy_descriptions}

The payoff table shows your payoff for each combination of your strategy (rows) and the other player's strategy (columns):
"""
        
        # Format payoff table
        other_strategies = P2_STRATEGY_DESCRIPTIONS.keys() if isinstance(self, SimulatorAgent) else P1_STRATEGY_DESCRIPTIONS
        prompt += "\n   |" + " ".join(f"{s:12}" for s in other_strategies) + "\n"
        prompt += "---+" + "------------" * len(other_strategies) + "\n"

        for i, strat in enumerate(self.strategies):
            row = f"{strat:3}|"
            for j in range(len(other_strategies)):
                row += f"\t {str(self.payoffs[i][j])}"
            prompt += row + "\n"

        if self.simulation_type == 'simulate_and_best_response':
            prompt += f"""\nYou may also choose to simulate the other player, which is represented by the additional action 'simulate'. 
In this case, you will receive the other player's mixed strategy, and automatically play the best response to that strategy. However, this incurs a cost of {self.simulation_cost}"""
        if default_format:
            prompt += """\nProvide a probability distribution over your strategies and 'simulate' as a JSON object where keys are strategies and values sum to 1.0.
Example format: {"strategy1": 0.5, "strategy2": 0.3, "simulate": 0.2}"""
        prompt += "\nOnly provide the JSON object, without any additional text."
        return prompt

    def compute_best_response(self, payoff_matrix: np.ndarray, other_strategy: Dict[str, float], 
                         available_strategies: List[str]) -> Dict[str, float]:
        """
        Compute the best response strategy using linear programming.
        Returns a probability distribution over available strategies.
        """
        other_probs = np.array(list(other_strategy.values()))
        
        # Expected payoff for each pure strategy
        expected_payoffs = np.dot(payoff_matrix, other_probs)
        
        # Find the strategies that give maximum payoff
        max_payoff = np.max(expected_payoffs)
        best_strategies = np.isclose(expected_payoffs, max_payoff, rtol=1e-10)
        
        # Compute probabilities
        probs = best_strategies.astype(float) / np.sum(best_strategies)
        
        return dict(zip(available_strategies, probs))

class SimulatedAgent(RestrictedTrustAgent):
    def __init__(
        self,
        name: str, 
        model: ModelWrapper,
        strategies: List[str],
        payoffs: np.ndarray
    ):
        super().__init__(name, model, strategies, payoffs)

    def get_strategy_elicitation_prompt(self, default_format=True) -> str:
        """Generate prompt for model to elicit strategy distribution"""
        strategy_descriptions = "\n".join([f"- {s}: {desc}" for s, desc in self.strategy_descriptions.items()])
        
        prompt = f"""You are playing a restricted trust game as {self.name}. Your available strategies are:
{strategy_descriptions}

The payoff table shows your payoff for each combination of your strategy (rows) and the other player's strategy (columns):
"""
        
        # Format payoff table
        other_strategies = P2_STRATEGY_DESCRIPTIONS.keys() if isinstance(self, SimulatorAgent) else P1_STRATEGY_DESCRIPTIONS
        prompt += "\n   |" + " ".join(f"{s:12}" for s in other_strategies) + "\n"
        prompt += "---+" + "------------" * len(other_strategies) + "\n"
        
        for i, strat in enumerate(self.strategies):
            row = f"{strat:3}|"
            for j in range(len(other_strategies)):
                row += f"{str(self.payoffs[j][i])}\t"
            prompt += row + "\n"

        if default_format:
            prompt += """\nProvide a probability distribution over your strategies as a JSON object where keys are strategies and values sum to 1.0.
Example format: {"strategy1": 0.5, "strategy2": 0.5}"""
        return prompt


class RestrictedTrustGame:
    def __init__(
        self,
        p1_strategies: List[str],
        p2_strategies: List[str],
        p1_model: ModelWrapper,
        p2_model: ModelWrapper,
        payoffs: np.ndarray,
        simulation_cost: float,
        simulation_type: str
    ):
        self.p1_strategies = p1_strategies
        self.p2_strategies = p2_strategies

        self.p1_payoffs = payoffs[:, :, 0]
        self.p2_payoffs = payoffs[:, :, 1]
        self.simulation_cost = simulation_cost
        self.simulation_type = simulation_type
        self.history = []

        self.p1 = SimulatorAgent("P1", p1_model, p1_strategies, self.p1_payoffs, simulation_cost, simulation_type)
        self.p2 = SimulatedAgent("P2", p2_model, p2_strategies, self.p2_payoffs)

        assert simulation_type in ['simulate_and_best_response'], "Invalid simulation type."

    def simulate_round(self):
        p1_strategy = self.p1.get_mixed_strategy()
        p2_strategy = self.p2.get_mixed_strategy()

        p1_choice = np.random.choice(self.p1_strategies + ['simulate'], p=list(p1_strategy.values()))
        if p1_choice == 'simulate' and self.simulation_type == 'simulate_and_best_response':
            # p1 plays best response to p2's strategy
            p1_best_response = self.p1.compute_best_response(self.p1_payoffs, p2_strategy, self.p1_strategies)
            p1_move = np.random.choice(self.p1_strategies, p=list(p1_best_response.values()))
        else:
            p1_move = p1_choice

        p2_choice = np.random.choice(self.p2_strategies, p=list(p2_strategy.values()))

        p1_payoff = self.p1_payoffs[self.p1_strategies.index(p1_move), self.p2_strategies.index(p2_choice)]
        if p1_choice == "simulate":
            p1_payoff -= self.simulation_cost
        p2_payoff = self.p2_payoffs[self.p1_strategies.index(p1_move), self.p2_strategies.index(p2_choice)]

        self.history.append({
            'p1_choice': p1_choice,
            'p1_move': p1_move,
            'p2_choice': p2_choice,
            'p1_payoff': p1_payoff,
            'p2_payoff': p2_payoff,
            'p1_strategy': p1_strategy,
            'p2_strategy': p2_strategy,
            'p1_gain_from_simulating': self.gain_from_simulating(p2_strategy)
        })

    def simulate_rounds(self, num_rounds: int):
        for _ in tqdm(range(num_rounds), desc="Simulating rounds"):
            self.simulate_round()

    def reset_history(self):
        self.history = []

    def equilibrium_payoffs(self) -> Tuple[float, float]:
        x_payoffs = self.p1_payoffs
        y_payoffs = self.p2_payoffs
        M = 1000 # arbirary large constant

        m, n = x_payoffs.shape
        p_x = cp.Variable(m)
        p_y = cp.Variable(n)
        b_x = cp.Variable(m, boolean=True)
        b_y = cp.Variable(n, boolean=True)
        u_x = cp.Variable(1)
        u_y = cp.Variable(1)

        obj = cp.Maximize(u_x + u_y) # welfare
        constraints = [
            p_x >= 0,
            p_y >= 0,
            cp.sum(p_x) == 1,
            cp.sum(p_y) == 1,
            p_x <= b_x,
            p_y <= b_y,

            u_x >= x_payoffs @ p_y,
            u_y >= y_payoffs.T @ p_x,
            u_x - x_payoffs @ p_y <= M*(1 - b_x),
            u_y - y_payoffs.T @ p_x <= M*(1 - b_y)
        ]

        prob = cp.Problem(obj, constraints)
        prob.solve(solver='GLPK_MI')
        return(u_x.value, u_y.value)

    def gain_from_simulating(self, p2_strategy: Dict[str, float]) -> float:
        """
        Calculate the gain from simulating P2's strategy and playing the best response,
        compared to the equilibrium payoff without simulation.
        """
        # Convert p2_strategy dict to a probability vector
        p2_probs = np.array([p2_strategy.get(s, 0) for s in self.p2_strategies])
        
        # Compute best response to p2's strategy
        best_response_dict = self.p1.compute_best_response(self.p1_payoffs, p2_strategy, self.p1_strategies)
        best_response_probs = np.array([best_response_dict.get(s, 0) for s in self.p1_strategies])
        
        # Calculate payoff from best response
        best_response_payoff = best_response_probs @ self.p1_payoffs @ p2_probs
        
        # Get payoff without simulating

        eq_payoff, _ = self.equilibrium_payoffs()
        
        # Calculate gain (considering simulation cost)
        gain = best_response_payoff - eq_payoff.item() - self.simulation_cost
        return gain

    def get_expected_payoffs(self, p1_strategy: Dict[str, float], p2_strategy: Dict[str, float]) -> Tuple[float, float]:
        """
        Calculate expected payoffs given mixed strategies for both players.
        
        Args:
            p1_strategy: Dictionary mapping P1 strategies to probabilities
            p2_strategy: Dictionary mapping P2 strategies to probabilities
            
        Returns:
            Tuple of (P1 expected payoff, P2 expected payoff)
        """
        # Convert strategy dictionaries to probability vectors
        p1_probs = np.array([p1_strategy.get(s, 0) for s in self.p1_strategies])
        p2_probs = np.array([p2_strategy.get(s, 0) for s in self.p2_strategies])
        
        # Check if P1 uses simulation
        p1_sim_prob = p1_strategy.get('simulate', 0)
        
        if p1_sim_prob > 0 and self.simulation_type == 'simulate_and_best_response':
            # Calculate best response to P2's strategy
            best_response_dict = SimulatorAgent.compute_best_response(
                self.p1_payoffs, p2_strategy, self.p1_strategies)
            best_response_probs = np.array([best_response_dict.get(s, 0) for s in self.p1_strategies])
            
            # Combine regular strategy and best response based on simulation probability
            combined_p1_probs = (1 - p1_sim_prob) * p1_probs + p1_sim_prob * best_response_probs
            
            # Calculate expected payoffs
            p1_payoff = combined_p1_probs @ self.p1_payoffs @ p2_probs - p1_sim_prob * self.simulation_cost
            p2_payoff = combined_p1_probs @ self.p2_payoffs @ p2_probs
        else:
            # Calculate expected payoffs without simulation
            p1_payoff = p1_probs @ self.p1_payoffs @ p2_probs
            p2_payoff = p1_probs @ self.p2_payoffs @ p2_probs
        
        return (p1_payoff, p2_payoff)

    def get_game_summary(self) -> Dict:
        """Return summary statistics of the game outcome"""
        if not self.history:
            return {
                "rounds_played": 0,
                "average_payoffs": (0, 0),
                "simulation_frequency": 0,
                "strategy_frequencies": {
                    "P1": {s: 0 for s in self.p1_strategies},
                    "P2": {s: 0 for s in self.p2_strategies}
                },
                "strategy_probabilities": {
                    "P1": {s: 0 for s in self.p1_strategies + ['simulate']},
                    "P2": {s: 0 for s in self.p2_strategies}
                }
            }
        
        rounds_played = len(self.history)
        p1_payoffs = [round['p1_payoff'] for round in self.history]
        p2_payoffs = [round['p2_payoff'] for round in self.history]
        
        p1_choices = [round['p1_choice'] for round in self.history]  # Initial choices
        p1_moves = [round['p1_move'] for round in self.history]    # Actual moves after simulation
        p2_choices = [round['p2_choice'] for round in self.history]

        p1_strategies = [round['p1_strategy'] for round in self.history]
        p2_strategies = [round['p2_strategy'] for round in self.history]
        
        simulation_count = sum(1 for choice in p1_choices if choice == 'simulate')
        simulation_frequency = simulation_count / rounds_played if rounds_played > 0 else 0
        
        # Calculate average gain from simulating
        simulation_gains = [round.get('p1_gain_from_simulating', 0) for round in self.history]
        avg_simulation_gain = sum(simulation_gains) / len(simulation_gains)
        
        # Calculate average probability mass for each strategy (NEW)
        p1_avg_probabilities = {}
        for strategy in self.p1_strategies + ['simulate']:
            p1_avg_probabilities[strategy] = sum(strat.get(strategy, 0) for strat in p1_strategies) / rounds_played
            
        p2_avg_probabilities = {}
        for strategy in self.p2_strategies:
            p2_avg_probabilities[strategy] = sum(strat.get(strategy, 0) for strat in p2_strategies) / rounds_played
        
        summary = {
            "rounds_played": rounds_played,
            "average_payoffs": (
                sum(p1_payoffs) / rounds_played,
                sum(p2_payoffs) / rounds_played
            ),
            "simulation_frequency": simulation_frequency,
            "average_simulation_gain": (avg_simulation_gain),
            "strategy_frequencies": {
                "P1": {
                    "initial_choices": {
                        s: p1_choices.count(s) / rounds_played 
                        for s in self.p1_strategies + ['simulate']
                    },
                    "final_moves": {
                        s: p1_moves.count(s) / rounds_played
                        for s in self.p1_strategies
                    }
                },
                "P2": {
                    "choices": {
                        s: p2_choices.count(s) / rounds_played 
                        for s in self.p2_strategies
                    }
                }
            },
            "strategy_probabilities": {
                "P1": p1_avg_probabilities,
                "P2": p2_avg_probabilities
            },
            "all_p1_strategies": p1_strategies,
            "all_p2_strategies": p2_strategies
        }
        
        return summary

    def write_summary_to_csv(self, csv_path: str, args=None):
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
            "round_index",
            "p1_choice",
            "p1_move",
            "p2_choice",
            "p1_payoff",
            "p2_payoff",
            "p1_gain_from_simulating"
        ]
        
        # Add P1 strategy probability columns
        for strategy in self.p1_strategies + ['simulate']:
            columns.append(f"p1_prob_{strategy}")
            
        # Add P2 strategy probability columns
        for strategy in self.p2_strategies:
            columns.append(f"p2_prob_{strategy}")
            
        # Add command line arguments if provided
        if args:
            for arg in vars(args):
                columns.append(f"arg_{arg}")
        
        # Prepare data for writing
        rows = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for round_idx, round_data in enumerate(self.history):
            row = {
                "timestamp": timestamp,
                "round_index": round_idx,
                "p1_choice": round_data["p1_choice"],
                "p1_move": round_data["p1_move"],
                "p2_choice": round_data["p2_choice"],
                "p1_payoff": round_data["p1_payoff"],
                "p2_payoff": round_data["p2_payoff"],
                "p1_gain_from_simulating": round_data["p1_gain_from_simulating"]
            }
            
            # Add P1 strategy probabilities for this round
            for strategy in self.p1_strategies + ['simulate']:
                row[f"p1_prob_{strategy}"] = round_data["p1_strategy"].get(strategy, 0)
                
            # Add P2 strategy probabilities for this round
            for strategy in self.p2_strategies:
                row[f"p2_prob_{strategy}"] = round_data["p2_strategy"].get(strategy, 0)
                
            # Add command line arguments if provided
            if args:
                for arg, value in vars(args).items():
                    row[f"arg_{arg}"] = value
                    
            rows.append(row)
        
        # Write to CSV
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            
            # Write headers only if file is new
            if not file_exists:
                writer.writeheader()
                
            # Write all rows
            writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description='Simulate a Restricted Trust Game')
    
    # Model parameters
    parser.add_argument('--p1_model', type=str, required=True, help='Model name for Player 1')
    parser.add_argument('--p2_model', type=str, required=True, help='Model name for Player 2')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for model sampling')
    
    # Game parameters
    parser.add_argument("--payoff-matrix-path", type=str, default='test.json')
    parser.add_argument('--rounds', type=int, default=100, help='Number of rounds to simulate')
    parser.add_argument('--simulation_cost', type=float, default=0.5, help='Cost of simulation')
    parser.add_argument('--simulation_type', type=str, default='simulate_and_best_response', 
                        choices=['simulate_and_best_response'], help='Type of simulation')
    
    # Output parameters
    parser.add_argument('--csv_output', type=str, default='game_results.csv',
                        help='File to save game summary statistics as CSV')
    parser.add_argument('--verbose', action='store_true', help='Print detailed output')
    
    args = parser.parse_args()
    
    # Define game parameters
    p1_strategies = list(P1_STRATEGY_DESCRIPTIONS.keys())
    p2_strategies = list(P2_STRATEGY_DESCRIPTIONS.keys())
    
    # Initialize model wrappers properly
    p1_model = ModelWrapper.create(args.p1_model, temperature=args.temperature)
    p2_model = ModelWrapper.create(args.p2_model, temperature=args.temperature)

    # Load payoffs from file
    with open(args.payoff_matrix_path, 'r') as f:
        payoff_dict = json.load(f)

    # Convert to numpy array
    payoffs = np.zeros((len(p1_strategies), len(p2_strategies), 2))
    for i, p1_strat in enumerate(p1_strategies):
        for j, p2_strat in enumerate(p2_strategies):
            payoffs[i, j] = payoff_dict[p1_strat][p2_strategies[j]]
    
    # Create the game
    game = RestrictedTrustGame(
        p1_strategies=p1_strategies,
        p2_strategies=p2_strategies,
        p1_model=p1_model,
        p2_model=p2_model,
        payoffs=payoffs,
        simulation_cost=args.simulation_cost,
        simulation_type=args.simulation_type
    )
    
    # Run simulation
    if args.verbose:
        print(f"Simulating {args.rounds} rounds...")
    game.simulate_rounds(args.rounds)
    
    # Get results
    results = game.get_game_summary()
    
    # Output results
    if args.verbose:
        print(f"Results:")
        print(f"Rounds played: {results['rounds_played']}")
        print(f"Average payoffs: P1={results['average_payoffs'][0]:.2f}, P2={results['average_payoffs'][1]:.2f}")
        print(f"Simulation frequency: {results['simulation_frequency']:.2%}")
        print(f"Average gain from simulating: {results['average_simulation_gain']:.2f}")
        print(f"\nStrategy Frequencies:")
        print(f"P1 initial choices: {results['strategy_frequencies']['P1']['initial_choices']}")
        print(f"P1 final moves: {results['strategy_frequencies']['P1']['final_moves']}")
        print(f"P2 choices: {results['strategy_frequencies']['P2']['choices']}")
        print(f"\nAverage Strategy Probabilities:")
        print(f"P1: {results['strategy_probabilities']['P1']}")
        print(f"P2: {results['strategy_probabilities']['P2']}")

    # Save summary to CSV file (NEW)
    game.write_summary_to_csv(args.csv_output, args)
    
    if args.verbose:
        print(f"Summary statistics saved to {args.csv_output}")
    
    return results

if __name__ == "__main__":
    main()