import json
from utils import parse_json
from typing import List, Dict, Tuple
from model_wrappers import ModelWrapper, Message
from tqdm import tqdm
import pandas as pd
import os
import argparse

class Agent:
    STRATEGIES = {
        "cooperative": "You aim to maximize mutual benefit while protecting against exploitation.",
        "defensive": "You prioritize minimizing potential losses over maximizing gains.",
        "tit_for_tat": "You begin by cooperating and then mirror your opponent's previous move.",
        "random": "You make decisions randomly without any particular strategy.",
        "always_cooperate": "You always cooperate regardless of circumstances.",
        "always_defect": "You always defect regardless of circumstances."
    }

    def __init__(self, model, strategy: str, name: str = None):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Strategy must be one of: {list(self.STRATEGIES.keys())}")
        
        self.model = model
        self.name = name or f"Agent_{strategy}"
        self.strategy = strategy
        self.last_opponent_move = None
        self.simulation_prompt = self._simulation_prompt_template()
        self.decision_prompt = self._decision_prompt_template()
    
    def _simulation_prompt_template(self) -> str:
        base_prompt = """You are playing an iterated prisoner's dilemma game. 
        On each turn, you must choose to either COOPERATE or DEFECT.
        On each turn, you may also choose to simulate the other player's move at a cost of 1 point.
        Note that you may also be simulated by the other player and you have no way of knowing if each game you are asked to play in is a simulation or a round of the actual game.
        
        Payoff structure:
        - If both players cooperate: Both get 3 points
        - If both players defect: Both get 1 point
        - If one cooperates and one defects: Defector gets 5 points, cooperator gets 0 points
        
        Your strategic disposition:
        {strategy}
        
        {simulation_prompt}
        
        Now determine if you will simulate the other player or continue to your move.
        Respond with either "SIMULATE" or "PLAY".
        """
        return base_prompt

    def _decision_prompt_template(self) -> str:
        base_prompt = """You are playing an iterated prisoner's dilemma game. 
        On each turn, you must choose to either COOPERATE or DEFECT.
        Note that you may be simulated by the other player and you have no way of knowing if each game you are asked to play in is a simulation or a round of the actual game.
        
        Payoff structure:
        - If both players cooperate: Both get 3 points
        - If both players defect: Both get 1 point
        - If one cooperates and one defects: Defector gets 5 points, cooperator gets 0 points
        
        Your strategic disposition:
        {strategy}
        
        {prediction_prompt}
        
        Respond with a JSON object containing:
        {response_format}
        """
        return base_prompt

class PrisonersDilemma:
    def __init__(
        self,
        model_name: str,
        agent1_strategy: str,
        agent2_strategy: str,
        sim_reasoning: bool = False,
        temperature: float = 0.7
    ):
        # Initialize model wrapper instances
        model1 = ModelWrapper.create(model_name, temperature=temperature)
        model2 = ModelWrapper.create(model_name, temperature=temperature)
        
        # Create agents
        self.agent1 = Agent(model1, agent1_strategy, "Agent1")
        self.agent2 = Agent(model2, agent2_strategy, "Agent2")
        self.sim_reasoning = sim_reasoning
        
        # Initialize results DataFrame
        self.results = pd.DataFrame(columns=[
            'round',
            'agent1_strategy',
            'agent2_strategy',
            'agent1_sim_decision',
            'agent2_sim_decision',
            'agent1_prediction',
            'agent2_prediction',
            'agent1_move',
            'agent2_move',
            'agent1_payoff',
            'agent2_payoff',
            'agent1_reasoning',
            'agent2_reasoning',
            'agent1_sim_move',
            'agent2_sim_move',
            'agent1_sim_correct',
            'agent2_sim_correct',
            'agent1_sim_reasoning',
            'agent2_sim_reasoning'
        ])
    
    def _create_round_simulation_prompt(self, agent: Agent, game_history: List[Dict] = None) -> str:
        """Create the complete prompt for a single round."""
        
        simulation_prompt = ""
            
        if game_history:
            history_str = "\nGame history:\n" + "\n".join(
                f"Round {i+1}: You {h['agent1_move']}, Opponent {h['agent2_move']}"
                for i, h in enumerate(game_history)
            )
            simulation_prompt += history_str
            
        return agent.simulation_prompt.format(
            strategy=Agent.STRATEGIES[agent.strategy],
            simulation_prompt=simulation_prompt
        )
    
    def _create_round_decision_prompt(self, agent: Agent, simulation_result: Dict = None, game_history: List[Dict] = None) -> str:
        """Create the complete prompt for a single round."""
        response_format = (
            '{"move": "COOPERATE/DEFECT", "reasoning": "your strategic reasoning"}' if not simulation_result else
            '{"prediction": "COOPERATE/DEFECT", "move": "COOPERATE/DEFECT", "reasoning": "your strategic reasoning"}'
        )
        
        prediction_prompt = ""
        if simulation_result:
            prediction_prompt = f"""
The simulated move by your opponent was: {simulation_result['move']}"""
            if self.sim_reasoning:
                prediction_prompt += f"""Your opponent's reasoning in simulation was: {simulation_result['reasoning']}"""
            prediction_prompt += """Based on this information and any game history, predict their move before making yours."""
            
        if game_history:
            history_str = "\nGame history:\n" + "\n".join(
                f"Round {i+1}: You {h['agent1_move']}, Opponent {h['agent2_move']}"
                for i, h in enumerate(game_history)
            )
            prediction_prompt += history_str
            
        return agent.decision_prompt.format(
            strategy=Agent.STRATEGIES[agent.strategy],
            prediction_prompt=prediction_prompt,
            response_format=response_format
        )
    
    def simulate_agent(self, agent, game_history, see_reasoning=False):
        """
        Simulate the opponent's move. Optionally, allow viewing their reasoning.
        """
        prompt = self._create_round_decision_prompt(agent, None, game_history)
        messages = [{"role": "user", "content": prompt}]
        response = agent.model.generate(messages)
        parsed = parse_json(response)
        if see_reasoning:
            return parsed
        return {"move": parsed["move"]}

    def play_round(self, round_num: int, game_history: List[Dict] = None) -> Dict:
        # Create simulation prompts for both agents
        prompt1 = self._create_round_simulation_prompt(self.agent1, game_history)
        prompt2 = self._create_round_simulation_prompt(self.agent2, game_history)
        
        # Get responses from both agents
        messages1: List[Message] = [{"role": "user", "content": prompt1}]
        messages2: List[Message] = [{"role": "user", "content": prompt2}]
        
        sim_decision1 = self.agent1.model.generate(messages1).strip().upper().replace("\n", "")
        sim_decision2 = self.agent2.model.generate(messages2).strip().upper().replace("\n", "")
        
        # Simulate opponent moves
        sim_result1 = self.simulate_agent(self.agent2, game_history) if "SIMULATE" in sim_decision1  else None
        sim_result2 = self.simulate_agent(self.agent1, game_history) if "SIMULATE" in sim_decision2 else None

        # Create decision prompts for both agents
        prompt1 = self._create_round_decision_prompt(self.agent1, sim_result2, game_history)
        prompt2 = self._create_round_decision_prompt(self.agent2, sim_result1, game_history)
        
        # Get responses from both agents
        messages1: List[Message] = [{"role": "user", "content": prompt1}]
        messages2: List[Message] = [{"role": "user", "content": prompt2}]

        response1 = self.agent1.model.generate(messages1)
        response2 = self.agent2.model.generate(messages2)

        # Parse responses
        try:
            parsed1 = parse_json(response1)
            parsed2 = parse_json(response2)
        except json.JSONDecodeError:
            # Fallback parsing
            parsed1 = {"move": "DEFECT", "prediction": "DEFECT"}
            parsed2 = {"move": "DEFECT", "prediction": "DEFECT"}
        
        # Calculate payoffs
        payoff1, payoff2 = self._calculate_payoffs(parsed1["move"], sim_decision1, parsed2["move"], sim_decision2)
        
        # Create round result
        round_result = {
            'round': round_num,
            'agent1_strategy': self.agent1.strategy,
            'agent2_strategy': self.agent2.strategy,
            'agent1_sim_decision': sim_decision1,
            'agent2_sim_decision': sim_decision2,
            'agent1_prediction': parsed1.get("prediction", ""),
            'agent2_prediction': parsed2.get("prediction", ""),
            'agent1_move': parsed1["move"],
            'agent2_move': parsed2["move"],
            'agent1_payoff': payoff1,
            'agent2_payoff': payoff2,
            'agent1_reasoning': parsed1.get("reasoning", ""),
            'agent2_reasoning': parsed2.get("reasoning", ""),
        }
        
        if sim_result1:
            round_result.update({
                'agent1_sim_move': sim_result1.get("move"),
                'agent1_sim_correct': sim_result1.get("move") == parsed2["move"]
            })
            if "reasoning" in sim_result1:
                round_result['agent1_sim_reasoning'] = sim_result1["reasoning"]
        if sim_result2:
            round_result.update({
                'agent2_sim_move': sim_result2.get("move"),
                'agent2_sim_correct': sim_result2.get("move") == parsed1["move"]
            })
            if "reasoning" in sim_result2:
                round_result['agent2_sim_reasoning'] = sim_result2["reasoning"]
        
        # Add to results DataFrame
        self.results.loc[len(self.results)] = round_result
        return round_result
    
    def _calculate_payoffs(self, move1: str, sim_decision1: str, move2: str, sim_decision2: str) -> Tuple[int, int]:
        payoff_matrix = {
            ('COOPERATE', 'COOPERATE'): (3, 3),
            ('COOPERATE', 'DEFECT'): (0, 5),
            ('DEFECT', 'COOPERATE'): (5, 0),
            ('DEFECT', 'DEFECT'): (1, 1)
        }
        payoff = payoff_matrix[(move1, move2)]
        if sim_decision1 == "SIMULATE":
            payoff = (payoff[0] - 1, payoff[1])
        if sim_decision2 == "SIMULATE":
            payoff = (payoff[0], payoff[1] - 1)
        return payoff
    
    def play_game(self, num_rounds: int) -> pd.DataFrame:
        game_history = []
        for round_num in tqdm(range(num_rounds), desc="Playing rounds"):
            round_result = self.play_round(round_num + 1, game_history)
            game_history.append(round_result)
        
        return self.results

def main():
    parser = argparse.ArgumentParser(description='Run Prisoner\'s Dilemma simulation with LLMs')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='Name of the LLM to use')
    parser.add_argument('--rounds', type=int, default=5,
                        help='Number of rounds to play')
    parser.add_argument('--agent1', type=str, default='cooperative',
                        choices=Agent.STRATEGIES.keys(),
                        help='Strategy for Agent 1')
    parser.add_argument('--agent2', type=str, default='defensive',
                        choices=Agent.STRATEGIES.keys(),
                        help='Strategy for Agent 2')
    parser.add_argument('--simreasoning', action='store_true',
                        help='Enable seeing reasoning in simulations')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for LLM sampling')
    parser.add_argument('--output', type=str, default='prisoner_dilemma_results.csv', help='Output file for results CSV')
    
    args = parser.parse_args()
    
    game = PrisonersDilemma(
        model_name=args.model,
        agent1_strategy=args.agent1,
        agent2_strategy=args.agent2,
        sim_reasoning=args.simreasoning,
        temperature=args.temperature
    )
    
    results = game.play_game(args.rounds)

    # Print summary statistics
    print("\nGame Summary:")
    print(f"Total rounds: {len(results)}")
    print("\nCooperation rates:")
    print(f"Agent 1 ({args.agent1}): {(results['agent1_move'] == 'COOPERATE').mean():.2%}")
    print(f"Agent 2 ({args.agent2}): {(results['agent2_move'] == 'COOPERATE').mean():.2%}")
    
    print("\nSimulation faithfulness:")
    print(f"Agent 1: {results['agent1_sim_correct'].mean():.2%}")
    print(f"Agent 2: {results['agent1_sim_correct'].mean():.2%}")
    
    # Save results
    if not os.path.exists('data'):
        os.makedirs('data')
    if os.path.exists(f'data/{args.output}'):
        df = pd.read_csv(f'data/{args.output}')
        results = pd.concat([df, results])
    results.to_csv(f'data/{args.output}', index=False)
    
    print(f"\nResults saved to data/{args.output}")

if __name__ == "__main__":
    main()