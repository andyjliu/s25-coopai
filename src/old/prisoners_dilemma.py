import json
from utils import parse_json
from typing import List, Dict, Tuple, Optional
from model_wrappers import ModelWrapper, Message
from tqdm import tqdm
import pandas as pd
import os
import random
import argparse


class Agent:
    """Base class for agents in the prisoner's dilemma game."""
    
    def __init__(self, name: str = None):
        """Initialize the base agent.
        
        Args:
            name: Optional name for the agent
        """
        self.name = name or f"Agent_{id(self)}"
    
    def next_move(self, game_history: List[Dict] = None) -> Dict:
        """Determine the next move based on game history.
        
        Args:
            game_history: List of past game rounds
            
        Returns:
            Dict containing move and other optional fields
        """
        # Base implementation returns a random move with empty strings for other fields
        return {
            "move": random.choice(["COOPERATE", "DEFECT"]),
            "prediction": "",
            "reasoning": ""
        }


class LLMAgent(Agent):
    """LLM-based agent in the prisoner's dilemma game."""
    
    # Dictionary of available strategies and their corresponding prompts
    STRATEGIES = {
        "cooperative": """You aim to maximize mutual benefit while protecting against exploitation.""",
            
        "defensive": """You prioritize minimizing potential losses over maximizing gains.""",
            
        "tit_for_tat": """You begin by cooperating and then mirror your opponent's previous move.""",
            
        "random": """You make decisions randomly without any particular strategy.""",
            
        "always_cooperate": """You always cooperate regardless of circumstances.""",
            
        "always_defect": """You always defect regardless of circumstances.""",
        
        "self_generated": """You will develop your own strategy based on your understanding of game theory and the prisoner's dilemma.""",
    
        "fictitious_play": """You track the frequency of your opponent's past choices and respond optimally to those frequencies. For example, if your opponent has chosen 'cooperate' 70 percent of the time and 'defect' 30 percent of the time in past rounds, you'll select the action that gives you the best expected payoff against this mixed strategy. You continuously update these frequencies after each round as new information becomes available."""
    }
    
    def __init__(
        self,
        model: ModelWrapper,
        strategy: str,
        name: str = None,
        self_generated_strategy: str = None
    ):
        """Initialize the LLM agent.
        
        Args:
            model: The LLM wrapper to use for generating responses
            strategy: Strategy key from STRATEGIES dict
            name: Optional name for the agent
            self_generated_strategy: Custom strategy text (for self_generated)
        """
        super().__init__(name)
        
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Strategy must be one of: {list(self.STRATEGIES.keys())}")
        
        self.model = model
        self.strategy = strategy
        self.self_generated_strategy = self_generated_strategy
        self.prompt_template = self._create_prompt_template()
        
    def _create_prompt_template(self) -> str:
        """Create the base prompt template for the agent."""
        base_prompt = """You are playing an iterated prisoner's dilemma game. 
        On each turn, you must choose to either COOPERATE or DEFECT.
        
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
    
    def generate_strategy(self) -> str:
        """Generates a custom strategy using the agent's model."""
        prompt = """You are playing an iterated prisoner's dilemma game.
        
        Payoff structure:
        - If both players cooperate: Both get 3 points
        - If both players defect: Both get 1 point
        - If one cooperates and one defects: Defector gets 5 points, cooperator gets 0 points
        
        Based on the description of the game, create a strategy that you will follow in the game.
        You will receive the opponent's strategy, as well as their past actions, if there are any.
        Given this, you should formulate a detailed description of how you will play initially, as well as how you will respond to your opponent's actions.
        
        Respond with a JSON object containing:
        {"strategy": "A clear description of your strategic approach (1-3 paragraphs)"}
        """
        
        messages: List[Message] = [{"role": "user", "content": prompt}]
        response = self.model.generate(messages)
        
        try:
            parsed = parse_json(response)
            return parsed.get("strategy", "Default to tit-for-tat with initial cooperation.")
        except json.JSONDecodeError:
            # Fallback if parsing fails
            return "Default to tit-for-tat with initial cooperation."
    
    def _create_round_prompt(self, opponent_strategy: str, predict_moves: bool, show_opponent_strategy: bool, opponent_strategy_text: str = None, game_history: List[Dict] = None) -> str:
        """Create the complete prompt for a single round.
        
        Args:
            opponent_strategy: Key of opponent's strategy
            predict_moves: Whether to enable move prediction
            show_opponent_strategy: Whether to reveal opponent's strategy
            opponent_strategy_text: Optional text of opponent's self-generated strategy
            game_history: List of past game rounds
            
        Returns:
            Formatted prompt string
        """
        response_format = (
            '{"move": "COOPERATE/DEFECT"}' if not predict_moves else
            '{"prediction": "COOPERATE/DEFECT", "move": "COOPERATE/DEFECT", "reasoning": "your strategic reasoning"}'
        )
        
        # Use the self-generated strategy if available
        strategy_text = self.self_generated_strategy if self.strategy == "self_generated" and self.self_generated_strategy else self.STRATEGIES[self.strategy]
        
        prediction_prompt = ""
        if predict_moves:
            # Only include opponent strategy info if show_opponent_strategy is True
            if show_opponent_strategy:
                # Determine opponent's strategy text
                if opponent_strategy == "self_generated" and opponent_strategy_text:
                    opponent_strat_text = opponent_strategy_text
                else:
                    opponent_strat_text = self.STRATEGIES[opponent_strategy]
                    
                prediction_prompt = f"""
Your opponent has the following strategic disposition:
{opponent_strat_text}
"""
            
            prediction_prompt += """
Based on this information and any game history, predict their move before making yours.
            """
            
        if game_history:
            history_str = "\nGame history:\n" + "\n".join(
                f"Round {i+1}: You {h['agent1_move'] if self.name == 'Agent1' else h['agent2_move']}, "
                f"Opponent {h['agent2_move'] if self.name == 'Agent1' else h['agent1_move']}"
                for i, h in enumerate(game_history)
            )
            prediction_prompt += history_str
            
        return self.prompt_template.format(
            strategy=strategy_text,
            prediction_prompt=prediction_prompt,
            response_format=response_format
        )
    
    def next_move(self, opponent_strategy: str, predict_moves: bool = False, show_opponent_strategy: bool = True, opponent_strategy_text: str = None, game_history: List[Dict] = None) -> Dict:
        """Generate the next move based on game history and opponent information.
        
        Args:
            opponent_strategy: Key of opponent's strategy
            predict_moves: Whether to enable move prediction
            show_opponent_strategy: Whether to reveal opponent's strategy
            opponent_strategy_text: Optional text of opponent's self-generated strategy
            game_history: List of past game rounds
            
        Returns:
            Dict containing move, prediction, and reasoning
        """
        # Create prompt
        prompt = self._create_round_prompt(opponent_strategy, predict_moves, show_opponent_strategy, opponent_strategy_text, game_history)
        
        # Get response from model
        messages: List[Message] = [{"role": "user", "content": prompt}]
        response = self.model.generate(messages)
        
        # Parse response
        try:
            parsed = parse_json(response)
        except json.JSONDecodeError:
            # Fallback parsing
            return {"move": "DEFECT", "prediction": "DEFECT", "reasoning": ""}
        
        # Ensure all fields are present
        result = {
            "move": parsed.get("move", "DEFECT"),
            "prediction": parsed.get("prediction", ""),
            "reasoning": parsed.get("reasoning", "")
        }
        
        return result


class FictitiousPlayAgent(Agent):
    """Agent that implements fictitious play strategy by tracking opponent's move frequencies
    and responding optimally."""
    
    def __init__(self, name: str = None, initial_belief: float = 0.5):
        """Initialize the fictitious play agent.
        
        Args:
            name: Optional name for the agent
            initial_belief: Initial belief about opponent's probability of cooperation (0-1)
        """
        super().__init__(name)
        self.strategy = "fictitious_play"
        self.cooperate_count = 0
        self.defect_count = 0
        # Start with a prior belief - default slightly optimistic
        self.initial_belief = initial_belief
    
    def next_move(self, game_history: List[Dict] = None) -> Dict:
        """Determine the next move based on opponent's history.
        
        Args:
            game_history: List of past game rounds
            
        Returns:
            Dict containing move and other info
        """
        # Update counts based on game history
        if game_history:
            for round_data in game_history:
                is_agent1 = self.name == "Agent1"
                opponent_move = round_data["agent2_move"] if is_agent1 else round_data["agent1_move"]
                
                if opponent_move == "COOPERATE":
                    self.cooperate_count += 1
                elif opponent_move == "DEFECT":
                    self.defect_count += 1
        
        # Calculate probability of opponent cooperation
        total_moves = self.cooperate_count + self.defect_count
        if total_moves == 0:
            # If no history, use initial belief
            p_cooperate = self.initial_belief
        else:
            p_cooperate = self.cooperate_count / total_moves
        
        # Calculate expected payoffs
        # If I cooperate: 3 * p_cooperate + 0 * (1-p_cooperate)
        # If I defect: 5 * p_cooperate + 1 * (1-p_cooperate)
        expected_cooperate = 3 * p_cooperate
        expected_defect = 5 * p_cooperate + 1 * (1 - p_cooperate)
        
        # Choose move with highest expected payoff
        if expected_cooperate > expected_defect:
            move = "COOPERATE"
            reasoning = f"Expected value of cooperation ({expected_cooperate:.2f}) exceeds defection ({expected_defect:.2f})"
        else:
            move = "DEFECT"
            reasoning = f"Expected value of defection ({expected_defect:.2f}) exceeds cooperation ({expected_cooperate:.2f})"
        
        return {
            "move": move,
            "prediction": "COOPERATE" if p_cooperate >= 0.5 else "DEFECT",
            "reasoning": reasoning
        }


class SimpleAgent(Agent):
    """Simple rule-based agent with predefined strategies."""
    
    def __init__(self, strategy: str, name: str = None):
        """Initialize a simple rule-based agent.
        
        Args:
            strategy: One of the predefined strategies
            name: Optional name for the agent
        """
        super().__init__(name)
        
        if strategy not in LLMAgent.STRATEGIES and strategy != "fictitious_play":
            raise ValueError(f"Strategy must be one of: {list(LLMAgent.STRATEGIES.keys())} or 'fictitious_play'")
        
        self.strategy = strategy
        self.last_opponent_move = None
    
    def next_move(self, game_history: List[Dict] = None) -> Dict:
        """Determine the next move based on strategy and game history.
        
        Args:
            game_history: List of past game rounds
            
        Returns:
            Dict containing move and other info
        """
        # Update last opponent move if we have history
        if game_history and len(game_history) > 0:
            last_round = game_history[-1]
            is_agent1 = self.name == "Agent1"
            self.last_opponent_move = last_round["agent2_move"] if is_agent1 else last_round["agent1_move"]
        
        # Determine move based on strategy
        if self.strategy == "always_cooperate":
            move = "COOPERATE"
        elif self.strategy == "always_defect":
            move = "DEFECT"
        elif self.strategy == "random":
            move = random.choice(["COOPERATE", "DEFECT"])
        elif self.strategy == "tit_for_tat":
            # Cooperate on first move, then mirror opponent
            if not self.last_opponent_move:
                move = "COOPERATE"
            else:
                move = self.last_opponent_move
        elif self.strategy == "defensive":
            # Defect more often, especially if opponent has defected
            if self.last_opponent_move == "DEFECT":
                move = "DEFECT"
            else:
                move = random.choices(["COOPERATE", "DEFECT"], weights=[0.3, 0.7])[0]
        elif self.strategy == "cooperative":
            # Cooperate more often, but occasionally defect
            if self.last_opponent_move == "DEFECT":
                move = random.choices(["COOPERATE", "DEFECT"], weights=[0.5, 0.5])[0]
            else:
                move = random.choices(["COOPERATE", "DEFECT"], weights=[0.8, 0.2])[0]
        else:
            # Fallback
            move = random.choice(["COOPERATE", "DEFECT"])
        
        return {
            "move": move,
            "prediction": "",
            "reasoning": ""
        }


class PrisonersDilemma:
    def __init__(
        self,
        model_name: str,
        agent1_strategy: str,
        agent2_strategy: str,
        predict_moves: bool = False,
        show_opponent_strategy: bool = True,
        self_generate_strategies: bool = False,
        temperature: float = 0.7,
        agent1_use_llm: bool = True,
        agent2_use_llm: bool = True
    ):
        """Initialize the prisoner's dilemma game.
        
        Args:
            model_name: Name of the LLM to use
            agent1_strategy: Strategy for first agent
            agent2_strategy: Strategy for second agent
            predict_moves: Whether to enable move prediction
            show_opponent_strategy: Whether to reveal opponent's strategy
            self_generate_strategies: Whether to generate custom strategies
            temperature: Temperature for LLM sampling
            agent1_use_llm: Whether to use LLM for Agent 1
            agent2_use_llm: Whether to use LLM for Agent 2
        """
        self.predict_moves = predict_moves
        self.show_opponent_strategy = show_opponent_strategy
        self.agent1_use_llm = agent1_use_llm
        self.agent2_use_llm = agent2_use_llm
        
        # Create model wrappers if using LLM
        if agent1_use_llm:
            model1 = ModelWrapper.create(model_name, temperature=temperature)
        if agent2_use_llm:
            model2 = ModelWrapper.create(model_name, temperature=temperature)
        
        # Generate strategies if enabled
        self.self_generate_strategies = self_generate_strategies
        agent1_strategy_text = None
        agent2_strategy_text = None
        
        if self_generate_strategies:
            if agent1_strategy == "self_generated" and agent1_use_llm:
                temp_agent1 = LLMAgent(model1, "self_generated", "TempAgent1")
                agent1_strategy_text = temp_agent1.generate_strategy()
                
            if agent2_strategy == "self_generated" and agent2_use_llm:
                temp_agent2 = LLMAgent(model2, "self_generated", "TempAgent2")
                agent2_strategy_text = temp_agent2.generate_strategy()
        
        # Create final agents
        if agent1_use_llm:
            self.agent1 = LLMAgent(model1, agent1_strategy, "Agent1", agent1_strategy_text)
        elif agent1_strategy == "fictitious_play":
            self.agent1 = FictitiousPlayAgent("Agent1")
        else:
            self.agent1 = SimpleAgent(agent1_strategy, "Agent1")    
            
        if agent2_use_llm:
            self.agent2 = LLMAgent(model2, agent2_strategy, "Agent2", agent2_strategy_text)
        elif agent2_strategy == "fictitious_play":
            self.agent2 = FictitiousPlayAgent("Agent2")
        else:
            self.agent2 = SimpleAgent(agent2_strategy, "Agent2")   
        
        # Initialize results DataFrame
        self.results = pd.DataFrame(columns=[
            'round',
            'agent1_strategy',
            'agent2_strategy',
            'agent1_strategy_text',
            'agent2_strategy_text',
            'agent1_move',
            'agent2_move',
            'agent1_payoff',
            'agent2_payoff',
            'agent1_prediction',
            'agent2_prediction',
            'agent1_prediction_correct',
            'agent2_prediction_correct',
            'agent1_reasoning',
            'agent2_reasoning'
        ])
        
        # Store the generated strategies in the instance
        self.agent1_strategy_text = agent1_strategy_text
        self.agent2_strategy_text = agent2_strategy_text
    
    def play_round(self, round_num: int, game_history: List[Dict] = None) -> Dict:
        """Play a single round of the game.
        
        Args:
            round_num: Current round number
            game_history: List of past game rounds
            
        Returns:
            Dict with round results
        """
        # Get moves from agents
        # Agent 1
        if self.agent1_use_llm:
            # For LLM agents, we need to pass the opponent strategy info
            result1 = self.agent1.next_move(
                opponent_strategy=self.agent2.strategy,
                predict_moves=self.predict_moves,
                show_opponent_strategy=self.show_opponent_strategy,
                opponent_strategy_text=self.agent2_strategy_text,
                game_history=game_history
            )
        else:
            # For non-LLM agents, just pass the game history
            result1 = self.agent1.next_move(game_history)
        
        # Agent 2
        if self.agent2_use_llm:
            # For LLM agents, we need to pass the opponent strategy info
            result2 = self.agent2.next_move(
                opponent_strategy=self.agent1.strategy,
                predict_moves=self.predict_moves,
                show_opponent_strategy=self.show_opponent_strategy,
                opponent_strategy_text=self.agent1_strategy_text,
                game_history=game_history
            )
        else:
            # For non-LLM agents, just pass the game history
            result2 = self.agent2.next_move(game_history)
        
        # Calculate payoffs
        payoff1, payoff2 = self._calculate_payoffs(result1["move"], result2["move"])
        
        # Create round result
        round_result = {
            'round': round_num,
            'agent1_strategy': self.agent1.strategy,
            'agent2_strategy': self.agent2.strategy,
            'agent1_strategy_text': self.agent1_strategy_text if hasattr(self.agent1, 'strategy') and self.agent1.strategy == "self_generated" else None,
            'agent2_strategy_text': self.agent2_strategy_text if hasattr(self.agent2, 'strategy') and self.agent2.strategy == "self_generated" else None,
            'agent1_move': result1["move"],
            'agent2_move': result2["move"],
            'agent1_payoff': payoff1,
            'agent2_payoff': payoff2,
            'agent1_reasoning': result1.get("reasoning", ""),
            'agent2_reasoning': result2.get("reasoning", "")
        }
        
        if self.predict_moves:
            round_result.update({
                'agent1_prediction': result1.get("prediction", ""),
                'agent2_prediction': result2.get("prediction", ""),
                'agent1_prediction_correct': result1.get("prediction") == result2["move"] if result1.get("prediction") else None,
                'agent2_prediction_correct': result2.get("prediction") == result1["move"] if result2.get("prediction") else None
            })
        
        # Add to results DataFrame
        self.results.loc[len(self.results)] = round_result
        return round_result
    
    def _calculate_payoffs(self, move1: str, move2: str) -> Tuple[int, int]:
        """Calculate payoffs based on the moves of both players.
        
        Args:
            move1: Move of the first player
            move2: Move of the second player
            
        Returns:
            Tuple of (payoff1, payoff2)
        """
        payoff_matrix = {
            ('COOPERATE', 'COOPERATE'): (3, 3),
            ('COOPERATE', 'DEFECT'): (0, 5),
            ('DEFECT', 'COOPERATE'): (5, 0),
            ('DEFECT', 'DEFECT'): (1, 1)
        }
        return payoff_matrix[(move1, move2)]
    
    def play_game(self, num_rounds: int) -> pd.DataFrame:
        """Play multiple rounds of the game.
        
        Args:
            num_rounds: Number of rounds to play
            
        Returns:
            DataFrame with all round results
        """
        game_history = []
        for round_num in tqdm(range(num_rounds), desc="Playing rounds"):
            round_result = self.play_round(round_num + 1, game_history)
            game_history.append(round_result)
        
        return self.results

def main():
    parser = argparse.ArgumentParser(description='Run Prisoner\'s Dilemma simulation with agents')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='Name of the LLM to use')
    parser.add_argument('--rounds', type=int, default=5,
                        help='Number of rounds to play')
    # Update strategy choices to include fictitious_play
    strategy_choices = list(LLMAgent.STRATEGIES.keys())
    parser.add_argument('--agent1', type=str, default='cooperative',
                        choices=strategy_choices,
                        help='Strategy for Agent 1')
    parser.add_argument('--agent2', type=str, default='defensive',
                        choices=strategy_choices,
                        help='Strategy for Agent 2')
    parser.add_argument('--predict', action='store_true',
                        help='Enable move prediction')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for LLM sampling')
    parser.add_argument('--output', type=str, default='prisoner_dilemma_results.csv',
                        help='Output file for results CSV')
    # Replace single use-llm flag with separate flags for each agent
    parser.add_argument('--agent1-use-llm', action='store_true', 
                        help='Use LLM for Agent 1')
    parser.add_argument('--agent2-use-llm', action='store_true',
                        help='Use LLM for Agent 2')
    # Add new argument for showing opponent strategy
    parser.add_argument('--show-opponent-strategy', action='store_true',
                        help='Show opponent strategy to agents')
    
    args = parser.parse_args()
    
    show_opponent_strategy = args.show_opponent_strategy
    
    # Automatically enable self-generated strategies if either agent uses them
    self_generate_strategies = args.agent1 == "self_generated" or args.agent2 == "self_generated"
    
    # LLM is required for self-generated strategies
    if args.agent1 == "self_generated":
        args.agent1_use_llm = True
    if args.agent2 == "self_generated":
        args.agent2_use_llm = True
    
    game = PrisonersDilemma(
        model_name=args.model,
        agent1_strategy=args.agent1,
        agent2_strategy=args.agent2,
        predict_moves=args.predict,
        show_opponent_strategy=show_opponent_strategy,
        self_generate_strategies=self_generate_strategies,
        temperature=args.temperature,
        agent1_use_llm=args.agent1_use_llm,
        agent2_use_llm=args.agent2_use_llm
    )
    
    # Print agent types
    print("\nAgent Types:")
    print(f"Agent 1 ({args.agent1}): {'LLM' if args.agent1_use_llm else 'Rule-based'}")
    print(f"Agent 2 ({args.agent2}): {'LLM' if args.agent2_use_llm else 'Rule-based'}")
    print(f"Agents can see opponent's strategy: {'Yes' if show_opponent_strategy else 'No'}")
    print()
    
    # Print the generated strategies if applicable
    if args.agent1 == "self_generated" or args.agent2 == "self_generated":
        print("\nGenerated Strategies:")
        if args.agent1 == "self_generated" and game.agent1_strategy_text:
            print(f"Agent 1 Strategy: {game.agent1_strategy_text}")
        if args.agent2 == "self_generated" and game.agent2_strategy_text:
            print(f"Agent 2 Strategy: {game.agent2_strategy_text}")
        print()
    
    results = game.play_game(args.rounds)

    # Print summary statistics
    print("\nGame Summary:")
    print(f"Total rounds: {len(results)}")
    print("\nCooperation rates:")
    print(f"Agent 1 ({args.agent1}): {(results['agent1_move'] == 'COOPERATE').mean():.2%}")
    print(f"Agent 2 ({args.agent2}): {(results['agent2_move'] == 'COOPERATE').mean():.2%}")
    
    if args.predict:
        print("\nPrediction accuracy:")
        print(f"Agent 1: {results['agent1_prediction_correct'].mean():.2%}")
        print(f"Agent 2: {results['agent2_prediction_correct'].mean():.2%}")
    
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