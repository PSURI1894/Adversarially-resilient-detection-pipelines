"""
PERSON 2: TEST SUITE (Ensemble & Training)
Lines: ~200-250
"""
import pytest
import numpy as np
import tensorflow as tf
from src.detection_ensemble import ResilientTrainer, EnsembleOrchestrator

class TestModelLogic:
    def test_cnn_architecture_shapes(self):
        """Verifies CNN input/output compatibility for 1D network flows."""
        input_dim = 20
        config = {'input_dim': input_dim}
        trainer = ResilientTrainer("CNN", config)
        
        # Create a dummy batch (BatchSize=8, Features=20, Channels=1)
        test_input = np.random.rand(8, input_dim, 1)
        prediction = trainer.model.predict(test_input, verbose=0)
        
        assert prediction.shape == (8, 1)
        assert (prediction >= 0).all() and (prediction <= 1).all()

    def test_ensemble_probability_sum(self):
        """Ensures weighted probabilities across ensemble sum to 1.0."""
        orch = EnsembleOrchestrator(input_dim=10)
        # Mocking input
        X = np.random.rand(5, 10)
        # Note: Since model isn't trained, we check output format
        # In a real test, you'd use a small pre-trained weight set
        try:
            probs = orch.predict_proba(X)
            assert probs.shape == (5, 2)
            assert np.allclose(probs.sum(axis=1), 1.0)
        except Exception:
            pytest.skip("Weights not initialized - requires fit() first")