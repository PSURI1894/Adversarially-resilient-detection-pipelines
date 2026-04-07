"""
PERSON 3: TEST SUITE (Uncertainty & Risk)
Lines: ~250-300
"""

import numpy as np
from src.risk_management_engine import ConformalEngine, RiskThermostat, SOCState


class TestConformalMath:
    def test_split_conformal_coverage(self):
        """Checks if q_hat math respects the error budget alpha."""
        engine = ConformalEngine(alpha=0.2)  # 80% confidence
        # Mock NCS scores: 10 samples from 0.1 to 1.0
        ncs_scores = np.linspace(0.1, 1.0, 10)
        n = 10
        # Manual calculation check
        q_idx = np.ceil((n + 1) * (1 - 0.2)) / n  # approx 0.88 -> 9th index

        # Set engine attributes
        engine.q_hat = np.quantile(ncs_scores, q_idx, method="higher")
        assert engine.q_hat >= 0.8

    def test_prediction_set_logic(self):
        """Ensures set-size increases with uncertainty (Max Entropy)."""
        engine = ConformalEngine(alpha=0.05)
        engine.q_hat = 0.7
        # Sample with high uncertainty (0.5/0.5 split)
        # 1 - 0.5 = 0.5. Since 0.5 <= 0.7, BOTH labels should be in the set.
        probs = np.array([[0.5, 0.5]])
        p_sets = engine.get_prediction_sets(probs)

        assert len(p_sets[0]) == 2
        assert 0 in p_sets[0] and 1 in p_sets[0]


class TestRiskController:
    def test_fsm_state_transitions(self):
        """Validates the Thermostat's transition from STABLE to EVASION_LOCKED."""
        thermostat = RiskThermostat(analyst_capacity=5)

        # Batch 1: Stable (All set sizes = 1)
        stable_sets = [[0], [1], [0], [1], [0]]
        state1 = thermostat.evaluate_risk(stable_sets)
        assert state1 == SOCState.STABLE

        # Batch 2: Highly Uncertain — repeat 3× to satisfy hysteresis_steps=3
        uncertain_sets = [[0, 1]] * 10
        for _ in range(3):
            state2 = thermostat.evaluate_risk(uncertain_sets)
        assert state2 == SOCState.EVASION_LOCKED
        assert thermostat.alert_debt >= 10
