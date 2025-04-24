import json
import numpy as np
from scipy.optimize import linprog

def parse_json(response):
    try:
        left = response.index('{')
        right = response.rindex('}') + 1
        return json.loads(response[left:right])
    except (ValueError, json.JSONDecodeError) as e:
        print(f'Error parsing JSON: {e}')
        return {}

def gpus_needed(model_name: str) -> int:
    if "70b" in model_name.lower():
        return 4
    else:
        return 1
    
def solve_stackelberg_game(p1_payoffs, p2_payoffs):
    """
    Solve a Stackelberg game where Player 2 is the leader and Player 1 is the follower.
    
    Parameters:
    -----------
    p1_payoffs : 2D numpy array
        Payoff matrix for Player 1 (follower)
    p2_payoffs : 2D numpy array
        Payoff matrix for Player 2 (leader)
        
    Returns:
    --------
    p1_strategy : 1D numpy array
        Optimal mixed strategy for Player 1
    p2_strategy : 1D numpy array
        Optimal mixed strategy for Player 2
    p1_payoff : float
        Expected payoff for Player 1
    p2_payoff : float
        Expected payoff for Player 2
    """
    num_p1_actions = p1_payoffs.shape[0]
    num_p2_actions = p1_payoffs.shape[1]
    
    # We'll try every pure strategy of Player 1 and find the optimal commitment
    # strategy for Player 2 that induces Player 1 to play that pure strategy
    best_p2_payoff = float('-inf')
    best_p1_strategy = None
    best_p2_strategy = None
    
    for p1_action in range(num_p1_actions):
        # For Player 1 to play action p1_action, we need to ensure it's a best response
        # to Player 2's mixed strategy
        
        # Variables: Player 2's mixed strategy probabilities
        c = [-p2_payoffs[p1_action, j] for j in range(num_p2_actions)]  # Maximize P2's payoff (negative for minimize)
        
        # Constraint: For each alternative action of Player 1, the expected payoff
        # should be less than or equal to the payoff from playing p1_action
        A_ub = []
        b_ub = []
        
        for p1_alt_action in range(num_p1_actions):
            if p1_alt_action == p1_action:
                continue
                
            # Expected payoff difference between p1_action and p1_alt_action must be >= 0
            # p1_payoffs[p1_action, :] * p2_strategy - p1_payoffs[p1_alt_action, :] * p2_strategy >= 0
            constraint = [p1_payoffs[p1_alt_action, j] - p1_payoffs[p1_action, j] for j in range(num_p2_actions)]
            A_ub.append(constraint)
            b_ub.append(0)
        
        # Constraint: Player 2's strategy probabilities sum to 1
        A_eq = [np.ones(num_p2_actions)]
        b_eq = [1.0]
        
        # Constraint: Probabilities are non-negative
        bounds = [(0, 1) for _ in range(num_p2_actions)]
        
        try:
            result = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), 
                             A_eq=np.array(A_eq), b_eq=np.array(b_eq), 
                             bounds=bounds, method='highs')
            
            if result.success:
                p2_strategy = result.x
                p2_payoff = -result.fun  # Negate since we minimized the negative
                
                if p2_payoff > best_p2_payoff:
                    best_p2_payoff = p2_payoff
                    best_p2_strategy = p2_strategy
                    
                    # Player 1's strategy is a pure strategy
                    p1_strategy = np.zeros(num_p1_actions)
                    p1_strategy[p1_action] = 1.0
                    best_p1_strategy = p1_strategy
        except:
            # Skip if the linear program is infeasible
            continue
    
    if best_p1_strategy is None:
        raise ValueError("Failed to find a solution")
    
    # Calculate Player 1's expected payoff
    p1_payoff = np.dot(np.dot(best_p1_strategy, p1_payoffs), best_p2_strategy)
    
    return best_p1_strategy, best_p2_strategy, p1_payoff, best_p2_payoff