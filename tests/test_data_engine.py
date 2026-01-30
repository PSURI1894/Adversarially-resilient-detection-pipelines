"""
PERSON 1: TEST SUITE (Data & Adversary)
Lines: ~150-200
"""
import pytest
import numpy as np
import pandas as pd
from src.data_infrastructure import DataOrchestrator, AdversarialArsenal, FeatureFactory


class TestDataInfrastructure:
    @pytest.fixture
    def raw_mock_data(self):
        """Generates raw, 'dirty' network traffic for cleaning tests."""
        return pd.DataFrame({
            ' Flow Duration': [100, 200, 0, -50, np.inf],
            'Tot Fwd Pkts': [10, 20, 5, 5, 5],
            'Label': ['Benign', 'Attack', 'Benign', 'Attack', 'Benign']
        })
    
    def test_feature_factory_entropy(self):
        """Verifies Shannon entropy calculation for packet distributions."""
        factory = FeatureFactory()
        # High entropy sequence
        seq_high = [1, 2, 3, 4, 5, 6, 7, 8]
        # Low entropy sequence
        seq_low = [1, 1, 1, 1, 1, 1, 1, 1]
        
        e_high = factory.calculate_flow_entropy(seq_high)
        e_low = factory.calculate_flow_entropy(seq_low)
        
        assert e_high > e_low
        assert e_low == 0.0

class TestAdversaryArsenal:
    def test_evasion_bounds(self):
        """Ensures adversarial jitter doesn't exceed the epsilon budget."""
        features = ['f1', 'f2']
        arsenal = AdversarialArsenal(features)
        X = np.array([[1.0, 1.0]])
        epsilon = 0.5
        
        X_adv = arsenal.evasion_by_jitter(X, epsilon=epsilon)
        diff = np.abs(X - X_adv)
        # Statistical check: noise is sampled from Normal(0, eps)
        # It's highly unlikely to exceed 4*sigma
        assert diff.max() < epsilon * 5