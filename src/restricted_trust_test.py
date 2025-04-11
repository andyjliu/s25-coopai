import unittest
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, List

# Import the classes to test
from restricted_trust_with_simulation import (
    RestrictedTrustAgent, 
    SimulatorAgent, 
    SimulatedAgent,
    RestrictedTrustGame,
    P1_STRATEGY_DESCRIPTIONS,
    P2_STRATEGY_DESCRIPTIONS
)

class TestRestrictedTrustAgent(unittest.TestCase):
    def setUp(self):
        self.mock_model = Mock()
        self.strategies = ["cooperate", "defect"]
        self.payoffs = np.array([
            [[10, 5], [0, 0]],  # trust payoffs
            [[7, 3], [2, 2]],   # partial_trust payoffs
            [[5, 0], [5, 0]]    # walk_out payoffs
        ])
        
        # Setup the agent
        self.agent = SimulatedAgent(
            name="P2",
            model=self.mock_model,
            strategies=self.strategies,
            payoffs=self.payoffs
        )
        self.agent.strategy_descriptions = P2_STRATEGY_DESCRIPTIONS
        
    def test_get_strategy_elicitation_prompt(self):
        prompt = self.agent.get_strategy_elicitation_prompt()
        
        # Check the prompt contains all strategies
        for strategy in self.strategies:
            self.assertIn(strategy, prompt)
            
        # Check it contains the payoff table
        self.assertIn("payoff table", prompt)
        
        # Check it requests a JSON response
        self.assertIn("JSON object", prompt)
        
    @patch("utils.parse_json")
    def test_get_mixed_strategy_valid_response(self, mock_parse_json):
        # Mock a valid response from the model
        valid_distribution = {
            "cooperate": 0.5,
            "defect": 0.5
        }
        mock_parse_json.return_value = valid_distribution
        self.mock_model.generate.return_value = "mock_response"
        
        result = self.agent.get_mixed_strategy()
        
        # Check the result is the expected distribution
        self.assertEqual(result, valid_distribution)
        self.mock_model.generate.assert_called_once()
        
    @patch("utils.parse_json")
    def test_get_mixed_strategy_invalid_response_wrong_keys(self, mock_parse_json):
        # Mock an invalid response with wrong keys
        invalid_distribution = {
            "cooperate": 0.5,
            "defect": 0.5
        }
        mock_parse_json.return_value = invalid_distribution
        self.mock_model.generate.return_value = "mock_response"
        
        # Should return uniform distribution on error
        result = self.agent.get_mixed_strategy()
        expected = {
            "cooperate": 0.5,
            "defect": 0.5
        }
        
        for key in expected:
            self.assertAlmostEqual(result[key], expected[key], places=5)
        
    @patch("utils.parse_json")
    def test_get_mixed_strategy_invalid_response_wrong_sum(self, mock_parse_json):
        # Mock an invalid response that doesn't sum to 1
        invalid_distribution = {
            "cooperate": 0.5,
            "defect": 0.7
        }
        mock_parse_json.return_value = invalid_distribution
        self.mock_model.generate.return_value = "mock_response"
        
        # Should return uniform distribution on error
        result = self.agent.get_mixed_strategy()
        expected = {
            "cooperate": 0.5,
            "defect": 0.5
        }
        
        for key in expected:
            self.assertAlmostEqual(result[key], expected[key], places=5)


class TestSimulatorAgent(unittest.TestCase):
    def setUp(self):
        self.mock_model = Mock()
        self.strategies = ["trust", "partial_trust", "walk_out"]
        self.payoffs = np.array([
            [[10, 5], [0, 0]],  # trust payoffs
            [[7, 3], [2, 2]],   # partial_trust payoffs
            [[5, 0], [5, 0]]    # walk_out payoffs
        ])
        self.simulation_cost = 1.0
        self.simulation_type = "simulate_and_best_response"
        
        # Setup the agent
        self.agent = SimulatorAgent(
            name="P1",
            model=self.mock_model,
            strategies=self.strategies,
            payoffs=self.payoffs,
            simulation_cost=self.simulation_cost,
            simulation_type=self.simulation_type
        )
        self.agent.strategy_descriptions = P1_STRATEGY_DESCRIPTIONS
        
    def test_get_strategy_elicitation_prompt_with_simulation(self):
        prompt = self.agent.get_strategy_elicitation_prompt()
        
        # Check the prompt contains all strategies
        for strategy in self.strategies:
            self.assertIn(strategy, prompt)
            
        # Check it contains simulation information
        self.assertIn("simulate", prompt)
        self.assertIn(str(self.simulation_cost), prompt)
        
        # Check it mentions the simulation type details
        self.assertIn("best response", prompt.lower())
        
    @patch("utils.parse_json")
    def test_get_mixed_strategy_with_simulation(self, mock_parse_json):
        # Mock a valid response including simulation
        valid_distribution = {
            "trust": 0.2,
            "partial_trust": 0.3,
            "walk_out": 0.1,
            "simulate": 0.4
        }
        mock_parse_json.return_value = valid_distribution
        self.mock_model.generate.return_value = "mock_response"
        
        result = self.agent.get_mixed_strategy()
        
        # Check the result is the expected distribution
        self.assertEqual(result, valid_distribution)
        self.mock_model.generate.assert_called_once()

    def test_compute_best_response(self):
        # Test the static method to compute best response
        payoff_matrix = np.array([
            [10, 0],  # Strategy 1 payoffs
            [7, 2],   # Strategy 2 payoffs
            [5, 5]    # Strategy 3 payoffs
        ])
        
        # Case 1: Other player plays pure strategy 1
        other_strategy = {"cooperate": 1.0, "defect": 0.0}
        strategies = ["trust", "partial_trust", "walk_out"]
        result = SimulatorAgent.compute_best_response(payoff_matrix, other_strategy, strategies)
        
        # Best response should be "trust" (highest payoff 10)
        expected = {"trust": 1.0, "partial_trust": 0.0, "walk_out": 0.0}
        self.assertEqual(result, expected)
        
        # Case 2: Other player plays pure strategy 2
        other_strategy = {"cooperate": 0.0, "defect": 1.0}
        result = SimulatorAgent.compute_best_response(payoff_matrix, other_strategy, strategies)
        
        # Best response should be "walk_out" (highest payoff 5)
        expected = {"trust": 0.0, "partial_trust": 0.0, "walk_out": 1.0}
        self.assertEqual(result, expected)
        
        # Case 3: Other player plays mixed strategy
        other_strategy = {"cooperate": 0.5, "defect": 0.5}
        result = SimulatorAgent.compute_best_response(payoff_matrix, other_strategy, strategies)
        
        # Expected payoffs:
        # trust: 10*0.5 + 0*0.5 = 5
        # partial_trust: 7*0.5 + 2*0.5 = 4.5
        # walk_out: 5*0.5 + 5*0.5 = 5
        # Best response should be "trust" and "walk_out" with equal probability
        expected = {"trust": 0.5, "partial_trust": 0.0, "walk_out": 0.5}
        self.assertEqual(result, expected)
        

class TestRestrictedTrustGame(unittest.TestCase):
    def setUp(self):
        self.p1_strategies = ["trust", "partial_trust", "walk_out"]
        self.p2_strategies = ["cooperate", "defect"]
        
        self.mock_p1_model = Mock()
        self.mock_p2_model = Mock()
        
        # Payoff matrix for (P1, P2)
        self.payoffs = np.array([
            [[10, 5], [0, 0]],  # trust payoffs
            [[7, 3], [2, 2]],   # partial_trust payoffs
            [[5, 0], [5, 0]]    # walk_out payoffs
        ])
        
        self.simulation_cost = 1.0
        self.simulation_type = "simulate_and_best_response"
        
        # Create the game
        self.game = RestrictedTrustGame(
            p1_strategies=self.p1_strategies,
            p2_strategies=self.p2_strategies,
            p1_model=self.mock_p1_model,
            p2_model=self.mock_p2_model,
            payoffs=self.payoffs,
            simulation_cost=self.simulation_cost,
            simulation_type=self.simulation_type
        )
        
    def test_game_initialization(self):
        # Check the game was initialized correctly
        self.assertEqual(self.game.p1_strategies, self.p1_strategies)
        self.assertEqual(self.game.p2_strategies, self.p2_strategies)
        self.assertEqual(self.game.simulation_cost, self.simulation_cost)
        self.assertEqual(self.game.simulation_type, self.simulation_type)
        
        # Check the agents were created
        self.assertIsInstance(self.game.p1, SimulatorAgent)
        self.assertIsInstance(self.game.p2, SimulatedAgent)
        
        # Check history is empty initially
        self.assertEqual(self.game.history, [])
        
    @patch.object(SimulatorAgent, "get_mixed_strategy")
    @patch.object(SimulatedAgent, "get_mixed_strategy")
    def test_simulate_round_without_simulation(self, mock_p2_strategy, mock_p1_strategy):
        # Mock the strategy choices
        p1_strat = {
            "trust": 0.0,
            "partial_trust": 1.0,
            "walk_out": 0.0,
            "simulate": 0.0
        }
        p2_strat = {
            "cooperate": 1.0,
            "defect": 0.0
        }
        mock_p1_strategy.return_value = p1_strat
        mock_p2_strategy.return_value = p2_strat
        
        # Force choice for deterministic testing
        with patch("numpy.random.choice") as mock_choice:
            mock_choice.side_effect = ["partial_trust", "cooperate"]
            
            self.game.simulate_round()
        
        # Check the history was updated
        self.assertEqual(len(self.game.history), 1)
        round_data = self.game.history[0]
        
        self.assertEqual(round_data["p1_choice"], "partial_trust")
        self.assertEqual(round_data["p1_move"], "partial_trust")
        self.assertEqual(round_data["p2_choice"], "cooperate")
        self.assertEqual(round_data["p1_payoff"], 7)
        self.assertEqual(round_data["p2_payoff"], 3)
        self.assertEqual(round_data["p1_strategy"], p1_strat)
        self.assertEqual(round_data["p2_strategy"], p2_strat)
        
    @patch.object(SimulatorAgent, "get_mixed_strategy")
    @patch.object(SimulatedAgent, "get_mixed_strategy")
    @patch.object(SimulatorAgent, "compute_best_response")
    def test_simulate_round_with_simulation(self, mock_best_response, mock_p2_strategy, mock_p1_strategy):
        # Mock the strategy choices
        p1_strat = {
            "trust": 0.0,
            "partial_trust": 0.0,
            "walk_out": 0.0,
            "simulate": 1.0
        }
        p2_strat = {
            "cooperate": 1.0,
            "defect": 0.0
        }
        best_response = {"trust": 1.0, "partial_trust": 0.0, "walk_out": 0.0}
        
        mock_p1_strategy.return_value = p1_strat
        mock_p2_strategy.return_value = p2_strat
        mock_best_response.return_value = best_response
        
        # Force choice for deterministic testing
        with patch("numpy.random.choice") as mock_choice:
            mock_choice.side_effect = ["simulate", "cooperate"]
            
            self.game.simulate_round()
        
        # Check the history was updated
        self.assertEqual(len(self.game.history), 1)
        round_data = self.game.history[0]
        
        self.assertEqual(round_data["p1_choice"], "simulate")
        self.assertEqual(round_data["p2_choice"], "cooperate")
        # Verify simulation cost was applied
        self.assertEqual(round_data["p1_payoff"], 10 - self.simulation_cost)  # Trust payoff - sim cost
        
    def test_reset_history(self):
        # Add some mock history
        self.game.history = [{"mock": "data"}]
        
        # Reset history
        self.game.reset_history()
        
        # Check history is empty
        self.assertEqual(self.game.history, [])
        
    def test_get_expected_payoffs_without_simulation(self):
        # P1 plays pure "partial_trust"
        p1_strat = {
            "trust": 0.0,
            "partial_trust": 1.0,
            "walk_out": 0.0
        }
        
        # P2 plays pure "cooperate"
        p2_strat = {
            "cooperate": 1.0,
            "defect": 0.0
        }
        
        p1_payoff, p2_payoff = self.game.get_expected_payoffs(p1_strat, p2_strat)
        
        # Expected payoffs for (partial_trust, cooperate)
        self.assertEqual(p1_payoff, 7)
        self.assertEqual(p2_payoff, 3)
        
    @patch.object(SimulatorAgent, "compute_best_response")
    def test_get_expected_payoffs_with_simulation(self, mock_best_response):
        # P1 plays pure "simulate"
        p1_strat = {
            "trust": 0.0,
            "partial_trust": 0.0,
            "walk_out": 0.0,
            "simulate": 1.0
        }
        
        # P2 plays pure "cooperate"
        p2_strat = {
            "cooperate": 1.0,
            "defect": 0.0
        }
        
        # Best response to "cooperate" is "trust"
        best_response = {"trust": 1.0, "partial_trust": 0.0, "walk_out": 0.0}
        mock_best_response.return_value = best_response
        
        p1_payoff, p2_payoff = self.game.get_expected_payoffs(p1_strat, p2_strat)
        
        # Expected payoffs: (trust, cooperate) - simulation cost
        self.assertEqual(p1_payoff, 10 - self.simulation_cost)
        self.assertEqual(p2_payoff, 5)
        
    def test_simulate_multiple_rounds(self):
        with patch.object(RestrictedTrustGame, "simulate_round") as mock_simulate:
            self.game.simulate_rounds(5)
            self.assertEqual(mock_simulate.call_count, 5)
    
    @patch.object(RestrictedTrustGame, "equilibrium_payoffs")
    def test_gain_from_simulating(self, mock_equilibrium):
        # Mock equilibrium payoffs
        mock_equilibrium.return_value = (6.0, 2.0)
        
        # P2 plays pure "cooperate"
        p2_strat = {
            "cooperate": 1.0,
            "defect": 0.0
        }
        
        # For pure "cooperate", best response is "trust" with payoff 10
        gain = self.game.gain_from_simulating(p2_strat)
        
        # Expected gain: Best response payoff (10) - Equilibrium payoff (6) - Simulation cost (1)
        self.assertEqual(gain, 10 - 6 - 1)

    def test_get_game_summary_with_empty_history(self):
        # Test with empty history
        summary = self.game.get_game_summary()
        
        self.assertEqual(summary["rounds_played"], 0)
        self.assertEqual(summary["average_payoffs"], (0, 0))
        self.assertEqual(summary["simulation_frequency"], 0)
        
    @patch.object(SimulatorAgent, "get_mixed_strategy")
    @patch.object(SimulatedAgent, "get_mixed_strategy")
    def test_get_game_summary_with_history(self, mock_p2_strategy, mock_p1_strategy):
        # Set up some game history
        p1_strat = {"trust": 0.5, "partial_trust": 0.5, "walk_out": 0.0}
        p2_strat = {"cooperate": 0.5, "defect": 0.5}
        
        mock_p1_strategy.return_value = p1_strat
        mock_p2_strategy.return_value = p2_strat
        
        # Force deterministic choices for 2 rounds
        with patch("numpy.random.choice") as mock_choice:
            mock_choice.side_effect = ["trust", "cooperate", "partial_trust", "defect"]
            
            self.game.simulate_rounds(2)
        
        # Get summary
        summary = self.game.get_game_summary()
        
        self.assertEqual(summary["rounds_played"], 2)
        # Payoffs should be average of (10,5) and (2,2)
        self.assertEqual(summary["average_payoffs"], ((10+2)/2, (5+2)/2))
        self.assertEqual(summary["simulation_frequency"], 0)
        
        # Check strategy frequencies
        p1_choices = summary["strategy_frequencies"]["P1"]["initial_choices"]
        self.assertEqual(p1_choices["trust"], 0.5)
        self.assertEqual(p1_choices["partial_trust"], 0.5)
        
        p2_choices = summary["strategy_frequencies"]["P2"]["choices"]
        self.assertEqual(p2_choices["cooperate"], 0.5)
        self.assertEqual(p2_choices["defect"], 0.5)


if __name__ == "__main__":
    unittest.main()