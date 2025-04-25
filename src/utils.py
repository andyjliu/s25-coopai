import json
import numpy as np
import math
from typing import Dict, List, Tuple, Any, Optional
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

import numpy as np
import math
from typing import Dict, List, Tuple, Any, Optional

def validate_trust_game(payoff_matrix: Dict[str, Dict[str, List[float]]]) -> Tuple[bool, str]:
    """
    Validates that the provided payoff matrix satisfies all properties of a 
    generalized partial-trust game:
    
    1. P1 has a strategy for opting out of the game (walk out)
    2. Trust enables profits but is exploitable
    3. There is a straightforward hierarchy of trust
    4. No tiebreaking property
    
    Args:
        payoff_matrix (dict): Dictionary representing the payoff matrix
                             with structure {p1_strategy: {p2_strategy: [p1_payoff, p2_payoff]}}
    
    Returns:
        bool: True if all properties are satisfied, False otherwise
        str: Explanation message (especially if any property is violated)
    """
    # Extract strategies
    p1_strategies = list(payoff_matrix.keys())
    if not p1_strategies:
        return False, "Empty payoff matrix"
    
    p2_strategies = list(payoff_matrix[p1_strategies[0]].keys())
    if not p2_strategies:
        return False, "No P2 strategies in payoff matrix"
    
    # Verify P2 has exactly two strategies (Cooperate and Defect)
    if len(p2_strategies) != 2:
        return False, f"P2 must have exactly 2 strategies (Cooperate and Defect), but has {len(p2_strategies)}"
    
    # For readability, map the strategies to cooperate/defect
    # Assuming the first strategy is cooperate and the second is defect
    cooperate, defect = p2_strategies
    
    # Property 1: P1 has a strategy for opting out of the game
    # Find strategies that could be the opt-out option (WO)
    possible_wo = []
    for strategy in p1_strategies:
        if (payoff_matrix[strategy][cooperate][0] == 0 and 
            payoff_matrix[strategy][cooperate][1] == 0 and
            payoff_matrix[strategy][defect][0] == 0 and 
            payoff_matrix[strategy][defect][1] == 0):
            possible_wo.append(strategy)
    
    if not possible_wo:
        return False, "Property 1 violation: No 'Walk Out' strategy found with payoffs (0,0) for both P2 strategies"
    
    walk_out = possible_wo[0]  # Use the first found WO strategy
    trust_strategies = [s for s in p1_strategies if s != walk_out]
    
    # Property 2: Trust enables profits but is exploitable
    for trust in trust_strategies:
        # u_1(T, C) > u_1(WO) = 0
        if not (payoff_matrix[trust][cooperate][0] > 0):
            return False, f"Property 2 violation: Strategy {trust} does not satisfy u_1(T, C) > 0"
        
        # 0 > u_1(T, D)
        if not (0 > payoff_matrix[trust][defect][0]):
            return False, f"Property 2 violation: Strategy {trust} does not satisfy 0 > u_1(T, D)"
        
        # u_2(T, D) > u_2(T, C)
        if not (payoff_matrix[trust][defect][1] > payoff_matrix[trust][cooperate][1]):
            return False, f"Property 2 violation: Strategy {trust} does not satisfy u_2(T, D) > u_2(T, C)"
        
        # u_2(T, C) > u_2(WO) = 0
        if not (payoff_matrix[trust][cooperate][1] > 0):
            return False, f"Property 2 violation: Strategy {trust} does not satisfy u_2(T, C) > 0"
    
    # Property 3: There is a straightforward hierarchy of trust
    # 3a: For any two strategies T != T', we have u_1(T, C) != u_1(T', C)
    p1_coop_values = [payoff_matrix[s][cooperate][0] for s in trust_strategies]
    if len(p1_coop_values) != len(set(p1_coop_values)):
        return False, "Property 3a violation: Some trust strategies have the same u_1(T, C) value"
    
    # 3b: When u_1(T, C) > u_1(T', C), we also have specific inequalities
    for t1 in trust_strategies:
        for t2 in trust_strategies:
            if t1 == t2:
                continue
                
            u1_t1_c = payoff_matrix[t1][cooperate][0]
            u1_t2_c = payoff_matrix[t2][cooperate][0]
            
            if u1_t1_c > u1_t2_c:
                # Check u_2(T, C) > u_2(T', C)
                if not (payoff_matrix[t1][cooperate][1] > payoff_matrix[t2][cooperate][1]):
                    return False, "Property 3b violation"
                
                # Check u_1(T, D) < u_1(T', D)
                if not (payoff_matrix[t1][defect][0] < payoff_matrix[t2][defect][0]):
                    return False, "Property 3b violation"
                
                # Check u_2(T, D) > u_2(T', D)
                if not (payoff_matrix[t1][defect][1] > payoff_matrix[t2][defect][1]):
                    return False, "Property 3b violation"
    
    # Property 4: No tiebreaking property
    no_tiebreaking, message = validate_no_tiebreaking_property(payoff_matrix)
    if not no_tiebreaking:
        return False, message
    
    # All checks passed
    return True, "The payoff matrix satisfies all properties of a generalized partial-trust game"

def validate_no_tiebreaking_property(payoff_matrix):
    """
    Validates that the payoff matrix satisfies the no tiebreaking property:
    For any T, if a convex combination of s_1 and t_1 satisfies 
    u_1(T, sigma_2) = u_1(sigma_1, sigma_2) for all sigma_2, 
    it must also satisfy u_2(T, sigma_2) = u_2(sigma_1, sigma_2) for all sigma_2.
    
    Args:
        payoff_matrix (dict): Dictionary representing the payoff matrix
                             with structure {p1_strategy: {p2_strategy: [p1_payoff, p2_payoff]}}
    
    Returns:
        bool: True if the property is satisfied, False otherwise
        str: Explanation message if the property is violated
    """
    # Extract all P1 strategies
    p1_strategies = list(payoff_matrix.keys())
    
    # Extract all P2 strategies
    p2_strategies = list(payoff_matrix[p1_strategies[0]].keys())
    
    # For each target strategy T
    for T in p1_strategies:
        # For each pair of other strategies (s_1, t_1)
        for s_1 in p1_strategies:
            if s_1 == T:
                continue
                
            for t_1 in p1_strategies:
                if t_1 == T or t_1 == s_1:
                    continue
                
                # Set up a linear program to find lambda where:
                # For all sigma_2: u_1(lambda*s_1 + (1-lambda)*t_1, sigma_2) = u_1(T, sigma_2)
                coefficients = []
                constants = []
                
                for sigma_2 in p2_strategies:
                    # lambda*u_1(s_1, sigma_2) + (1-lambda)*u_1(t_1, sigma_2) = u_1(T, sigma_2)
                    # Rearranging: lambda*[u_1(s_1, sigma_2) - u_1(t_1, sigma_2)] = u_1(T, sigma_2) - u_1(t_1, sigma_2)
                    coef = payoff_matrix[s_1][sigma_2][0] - payoff_matrix[t_1][sigma_2][0]
                    const = payoff_matrix[T][sigma_2][0] - payoff_matrix[t_1][sigma_2][0]
                    
                    coefficients.append(coef)
                    constants.append(const)
                
                # Solve the system
                solution_exists, lambda_value = solve_linear_system(coefficients, constants)
                
                if solution_exists:
                    # Check if it also gives the same P2 utility
                    for sigma_2 in p2_strategies:
                        # Calculate P2's utility with the convex combination
                        combined_u2 = (lambda_value * payoff_matrix[s_1][sigma_2][1] + 
                                      (1 - lambda_value) * payoff_matrix[t_1][sigma_2][1])
                        
                        # Check if it equals P2's utility with T
                        if not math.isclose(combined_u2, payoff_matrix[T][sigma_2][1], abs_tol=1e-10):
                            return False, f"Property 4 violation: For strategies {T}, {s_1}, {t_1} with λ={lambda_value:.4f}, " \
                                         f"P1 utilities match but P2 utilities differ for P2 strategy {sigma_2}"
    
    # If we get here, the property is satisfied
    return True, "The payoff matrix satisfies the no tiebreaking property"

def solve_linear_system(coefficients, constants):
    """
    Solve the linear system coefficients * lambda = constants
    
    Returns:
        bool: Whether a solution exists in [0,1]
        float: The solution lambda if it exists, None otherwise
    """
    import numpy as np
    
    A = np.array(coefficients).reshape(-1, 1)
    b = np.array(constants)
    
    # Handle edge cases
    if len(A) == 0:
        return False, None
    
    # Solve the system
    try:
        # Check if system is consistent by examining rank
        augmented = np.column_stack((A, b))
        rank_A = np.linalg.matrix_rank(A)
        rank_aug = np.linalg.matrix_rank(augmented)
        
        if rank_A != rank_aug:
            # System is inconsistent
            return False, None
        
        # Solve the system
        if rank_A < A.shape[1]:  # Underdetermined system
            # For underdetermined systems, find the least-squares solution
            lambda_value = np.linalg.lstsq(A, b, rcond=None)[0][0]
        else:
            # Use direct solver for well-determined and overdetermined systems
            if A.shape[0] > A.shape[1]:  # Overdetermined
                lambda_value = np.linalg.lstsq(A, b, rcond=None)[0][0]
            else:  # Well-determined
                lambda_value = np.linalg.solve(A, b)[0]
        
        # Check if solution is valid (approximately in [0,1])
        if -1e-10 <= lambda_value <= 1 + 1e-10:
            # Clamp to [0,1] in case of small numerical errors
            lambda_value = max(0, min(1, lambda_value))
            
            # For all equations, check if they are approximately satisfied
            for i in range(len(coefficients)):
                expected = constants[i]
                actual = coefficients[i] * lambda_value
                
                if not math.isclose(expected, actual, abs_tol=1e-8):
                    return False, None
            
            return True, lambda_value
        
    except np.linalg.LinAlgError:
        # Singular matrix, no unique solution
        pass
    
    return False, None