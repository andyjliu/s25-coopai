import json
from typing import List, Dict, Tuple, Optional
from utils import parse_json
from model_wrappers import ModelWrapper, Message
from tqdm import tqdm
import pandas as pd
import os
import argparse

class Agent:
    """Represents a single LLM agent in the restricted trust game."""
    
    # Dictionary of available strategies and their corresponding prompts
    STRATEGIES = {
        "trusting": "You tend to trust others and believe in doing things that are mutually beneficial.",
        "rational": "You tend to be rational and do things that lead to the most beneficial outcome in the long run.",
        "greedy": "You tend to be selfish and do things that are most beneficial to you.",
        "reciprocal": "You tend to reciprocate the actions of others, whether good or bad.",
        "unpredictable": "You tend to be unpredictable and do things that are least expected.",
        "shortsighted": "You tend to be short-term focused and do things that are most beneficial in the short term."
    }
    
    def __init__(
        self,
        model: ModelWrapper,
        strategy: str,
        role: str,  # 'investor' or 'trustee'
        name: str = None
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Strategy must be one of: {list(self.STRATEGIES.keys())}")
        if role not in ['investor', 'trustee']:
            raise ValueError("Role must be either 'investor' or 'trustee'")
        
        self.model = model
        self.strategy = strategy
        self.role = role
        self.name = name or f"Agent_{strategy}_{role}"
        self.prompt_template = self._create_prompt_template()
        
    def _create_prompt_template(self) -> str:
        if self.role == 'investor':
            base_prompt = """You are playing a restricted trust game as the investor. 
            You start with 10 tokens. You must decide how many tokens (0-10) to invest.
            Any amount you invest will be tripled before reaching the trustee.
            The trustee can then decide how much of the tripled amount to return to you.
            
            {model_awareness}
            
            Your strategic disposition:
            {strategy}
            
            {game_history}
            
            {prediction_prompt}
            
            Respond with a JSON object containing:
            {response_format}
            """
        else:  # trustee
            base_prompt = """You are playing a restricted trust game as the trustee.
            The investor has invested {investment} tokens, which has been tripled to {tripled_amount} tokens.
            You must decide how many tokens (0-{tripled_amount}) to return to the investor.
            
            {model_awareness}
            
            Your strategic disposition:
            {strategy}
            
            {game_history}
            
            {prediction_info}
            
            Respond with a JSON object containing:
            {{"reasoning": "your strategic reasoning (think step by step)",
              "return_amount": <number 0-{tripled_amount}>}}
            """
        
        return base_prompt

class RestrictedTrustGame:
    def __init__(
        self,
        model_name: str,
        investor_strategy: str,
        trustee_strategy: str,
        enable_predictions: bool = False,
        enable_model_awareness: bool = False,
        temperature: float = 0.7,
        initial_tokens: int = 10,
        multiplier: int = 3
    ):
        # Initialize model wrapper instances
        model1 = ModelWrapper.create(model_name, temperature=temperature)
        model2 = ModelWrapper.create(model_name, temperature=temperature)
        
        # Create agents
        self.investor = Agent(model1, investor_strategy, 'investor', "Investor")
        self.trustee = Agent(model2, trustee_strategy, 'trustee', "Trustee")
        
        self.initial_tokens = initial_tokens
        self.multiplier = multiplier
        self.enable_predictions = enable_predictions
        self.enable_model_awareness = enable_model_awareness
        
        # Initialize results DataFrame columns
        columns = [
            'round',
            'investor_strategy',
            'trustee_strategy',
            'investment',
            'tripled_amount',
            'return_amount',
            'investor_final',
            'trustee_final',
            'investor_reasoning',
            'trustee_reasoning'
        ]
        if enable_predictions:
            columns.extend(['predicted_return', 'prediction_error'])
            
        self.results = pd.DataFrame(columns=columns)
    
    def _create_round_prompt(self, agent: Agent, investment: int = None, predicted_return: int = None, game_history: List[Dict] = None) -> str:
        """Create the complete prompt for a single round."""
        # Prepare game history string
        history_str = ""
        if game_history:
            if self.enable_predictions:
                history_str = "\nGame history:\n" + "\n".join(
                    f"Round {i+1}: Investment = {h['investment']}, Predicted Return = {h['predicted_return']}, Actual Return = {h['return_amount']}"
                    for i, h in enumerate(game_history)
                )
            else:
                history_str = "\nGame history:\n" + "\n".join(
                    f"Round {i+1}: Investment = {h['investment']}, Return = {h['return_amount']}"
                    for i, h in enumerate(game_history)
                )
        
        # Prepare model awareness string
        model_awareness = ""
        if self.enable_model_awareness:
            if agent.role == 'investor':
                model_awareness = f"""Important: The trustee is another instance of the same language model as you, also following their own strategic disposition.
                The trustee's strategy is: {Agent.STRATEGIES[self.trustee.strategy]}"""
            else:
                model_awareness = f"""Important: The investor is another instance of the same language model as you, also following their own strategic disposition.
                The investor's strategy is: {Agent.STRATEGIES[self.investor.strategy]}"""
        
        # Prepare prediction-related content
        prediction_prompt = ""
        prediction_info = ""
        response_format = ""
        
        if agent.role == 'investor':
            if self.enable_predictions:
                prediction_prompt = """Think through this step by step:
                1. Consider the trustee's likely behavior
                2. Predict their likely return amount based on your investment
                3. Make your investment decision accordingly"""
                response_format = """{
                    "reasoning": "your strategic reasoning (think step by step)", 
                    "investment": <number 0-10>,
                    "predicted_return": <number indicating how many tokens you expect to receive back>
                }"""
            else:
                response_format = """{
                    "reasoning": "your strategic reasoning (think step by step)", 
                    "investment": <number 0-10>
                }"""
        else:  # trustee
            if self.enable_predictions:
                prediction_info = f"The investor predicted a return of: {predicted_return} tokens"
            
        if agent.role == 'investor':
            return agent.prompt_template.format(
                strategy=Agent.STRATEGIES[agent.strategy],
                model_awareness=model_awareness,
                game_history=history_str,
                prediction_prompt=prediction_prompt,
                response_format=response_format
            )
        else:  # trustee
            return agent.prompt_template.format(
                strategy=Agent.STRATEGIES[agent.strategy],
                model_awareness=model_awareness,
                investment=investment,
                tripled_amount=investment * self.multiplier,
                game_history=history_str,
                prediction_info=prediction_info
            )
    
    def play_round(self, round_num: int, game_history: List[Dict] = None) -> Dict:
        # Get investor's move
        investor_prompt = self._create_round_prompt(self.investor, game_history=game_history)
        messages1: List[Message] = [{"role": "user", "content": investor_prompt}]
        investor_response = self.investor.model.generate(messages1)
        
        try:
            investor_parsed = parse_json(investor_response)
            investment = min(max(0, int(investor_parsed["investment"])), self.initial_tokens)
            if self.enable_predictions:
                predicted_return = min(max(0, int(investor_parsed.get("predicted_return", 0))), investment * self.multiplier)
            else:
                predicted_return = None
        except (json.JSONDecodeError, KeyError, ValueError):
            investment = 0
            predicted_return = 0 if self.enable_predictions else None
            investor_parsed = {"reasoning": "Failed to parse response"}
        
        # Get trustee's move
        trustee_prompt = self._create_round_prompt(
            self.trustee, 
            investment=investment,
            predicted_return=predicted_return,
            game_history=game_history
        )
        messages2: List[Message] = [{"role": "user", "content": trustee_prompt}]
        trustee_response = self.trustee.model.generate(messages2)
        
        tripled_amount = investment * self.multiplier
        try:
            trustee_parsed = parse_json(trustee_response)
            return_amount = min(max(0, int(trustee_parsed["return_amount"])), tripled_amount)
        except (json.JSONDecodeError, KeyError, ValueError):
            return_amount = 0
            trustee_parsed = {"reasoning": "Failed to parse response"}
        
        # Calculate final payoffs
        investor_final = self.initial_tokens - investment + return_amount
        trustee_final = tripled_amount - return_amount

        print(investor_parsed, trustee_parsed)
        
        # Create round result
        round_result = {
            'round': round_num,
            'investor_strategy': self.investor.strategy,
            'trustee_strategy': self.trustee.strategy,
            'investment': investment,
            'tripled_amount': tripled_amount,
            'return_amount': return_amount,
            'investor_final': investor_final,
            'trustee_final': trustee_final,
            'investor_reasoning': investor_parsed.get("reasoning", ""),
            'trustee_reasoning': trustee_parsed.get("reasoning", "")
        }
        
        if self.enable_predictions:
            round_result.update({
                'predicted_return': predicted_return,
                'prediction_error': abs(predicted_return - return_amount)
            })
        
        # Add to results DataFrame
        self.results.loc[len(self.results)] = round_result
        return round_result
    
    def play_game(self, num_rounds: int) -> pd.DataFrame:
        game_history = []
        for round_num in tqdm(range(num_rounds), desc="Playing rounds"):
            round_result = self.play_round(round_num + 1, game_history)
            game_history.append(round_result)
        
        return self.results

def main():
    parser = argparse.ArgumentParser(description='Run Restricted Trust Game simulation with LLMs')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='Name of the LLM to use')
    parser.add_argument('--rounds', type=int, default=5,
                        help='Number of rounds to play')
    parser.add_argument('--investor', type=str, default='rational',
                        choices=Agent.STRATEGIES.keys(),
                        help='Strategy for Investor')
    parser.add_argument('--trustee', type=str, default='greedy',
                        choices=Agent.STRATEGIES.keys(),
                        help='Strategy for Trustee')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for LLM sampling')
    parser.add_argument('--enable-predictions', action='store_true',
                        help='Enable return amount predictions by investor')
    parser.add_argument('--enable-model-awareness', action='store_true',
                        help='Inform agents they are playing against same model type')
    parser.add_argument('--output', type=str, default='trust_game_results.csv',
                        help='Output file for results CSV')
    
    args = parser.parse_args()
    
    game = RestrictedTrustGame(
        model_name=args.model,
        investor_strategy=args.investor,
        trustee_strategy=args.trustee,
        temperature=args.temperature,
        enable_predictions=args.enable_predictions,
        enable_model_awareness=args.enable_model_awareness
    )
    
    results = game.play_game(args.rounds)

    # Print summary statistics
    print("\nGame Summary:")
    print(f"Total rounds: {len(results)}")
    print(f"\nAverage statistics:")
    print(f"Investment amount: {results['investment'].mean():.2f}")
    print(f"Return ratio: {(results['return_amount'] / results['tripled_amount']).mean():.2%}")
    
    if args.enable_predictions:
        print(f"Prediction accuracy: {results['prediction_error'].mean():.2f} tokens average error")
        
    print(f"Investor final tokens (avg): {results['investor_final'].mean():.2f}")
    print(f"Trustee final tokens (avg): {results['trustee_final'].mean():.2f}")
    
    # Save results
    os.makedirs('data', exist_ok=True)
    if os.path.exists(f'data/{args.output}'):
        df = pd.read_csv(f'data/{args.output}')
        results = pd.concat([df, results])
    results.to_csv(f'data/{args.output}', index=False)
    
    print(f"\nResults saved to data/{args.output}")

if __name__ == "__main__":
    main()