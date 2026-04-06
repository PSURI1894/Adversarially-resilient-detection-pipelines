"""
================================================================================
CONCEPT DRIFT DETECTION ENGINE
================================================================================
Implements multiple drift detection algorithms:
- ADWIN (Adaptive Windowing)
- Page-Hinkley
- Kolmogorov-Smirnov (KS) Test
- Maximum Mean Discrepancy (MMD)
================================================================================
"""

import numpy as np
from scipy import stats
from collections import deque
from typing import List, Dict, Any, Optional

class ADWINDetector:
    """
    Simplified Adaptive Windowing (ADWIN) for drift detection.
    Detects shifts in the mean of a stream.
    """
    def __init__(self, delta: float = 0.002, window_size: int = 1000):
        self.delta = delta
        self.window_size = window_size
        self.stream = deque(maxlen=window_size)
    
    def update(self, v: float) -> bool:
        self.stream.append(v)
        if len(self.stream) < self.window_size:
            return False
        
        # Split window into two parts and check for difference in mean
        # Simplified version: split at the middle
        mid = len(self.stream) // 2
        W1 = list(self.stream)[:mid]
        W2 = list(self.stream)[mid:]
        
        n1, n2 = len(W1), len(W2)
        mu1, mu2 = np.mean(W1), np.mean(W2)
        
        # Adwin threshold calculation (simplified)
        m = 1.0 / (1.0 / n1 + 1.0 / n2)
        epsilon = np.sqrt(1.0 / (2 * m) * np.log(4 / self.delta))
        
        return abs(mu1 - mu2) > epsilon

class PageHinkleyDetector:
    """
    Page-Hinkley test for detecting change points in a stream.
    Useful for monitoring error rates.
    """
    def __init__(self, delta: float = 0.005, lambda_threshold: float = 50, alpha: float = 0.9999):
        self.delta = delta
        self.lambda_threshold = lambda_threshold
        self.alpha = alpha
        self.x_mean = 0
        self.sum = 0
        self.n = 0
    
    def update(self, x: float) -> bool:
        self.n += 1
        self.x_mean = self.x_mean + (x - self.x_mean) / self.n
        self.sum = self.alpha * self.sum + (x - self.x_mean - self.delta)
        
        if self.sum > self.lambda_threshold:
            self.reset()
            return True
        return False
    
    def reset(self):
        self.sum = 0
        self.x_mean = 0
        self.n = 0

class KSDetector:
    """
    Kolmogorov-Smirnov test for distributional shift between reference and current data.
    """
    def __init__(self, reference_data: np.ndarray, alpha: float = 0.05):
        self.reference_data = reference_data
        self.alpha = alpha
    
    def detect(self, current_data: np.ndarray) -> bool:
        # Perform KS test per feature
        # If any feature shows significant shift, return True
        # For simplicity, we can aggregate or check major features
        drift_detected = False
        for i in range(self.reference_data.shape[1]):
            stat, p_val = stats.ks_2samp(self.reference_data[:, i], current_data[:, i])
            if p_val < self.alpha:
                drift_detected = True
                break
        return drift_detected

class MMDDetector:
    """
    Maximum Mean Discrepancy (MMD) for kernel-based two-sample testing.
    """
    def __init__(self, reference_data: np.ndarray, alpha: float = 0.05):
        self.reference_data = reference_data
        self.alpha = alpha
        self.ref_mean_emb = self._compute_mean_emb(reference_data)
        
    def _compute_mean_emb(self, data: np.ndarray) -> np.ndarray:
        # Simplified RBF kernel mean embedding
        return np.mean(data, axis=0)
    
    def detect(self, current_data: np.ndarray) -> bool:
        curr_mean_emb = self._compute_mean_emb(current_data)
        dist = np.linalg.norm(self.ref_mean_emb - curr_mean_emb)
        # Thresholding logic for MMD distance
        # For now, simplistic threshold based on reference variance
        threshold = np.std(self.reference_data) * 0.5
        return dist > threshold

class ConceptDriftEngine:
    """
    Multi-signal consensus drift detection engine.
    """
    def __init__(self, reference_features: np.ndarray):
        self.adwin = ADWINDetector()
        self.ph = PageHinkleyDetector()
        self.ks = KSDetector(reference_features)
        self.mmd = MMDDetector(reference_features)
        
        self.detectors = [self.adwin, self.ph, self.ks, self.mmd]
    
    def evaluate(self, current_batch: np.ndarray, prediction_errors: List[float]) -> bool:
        """
        Evaluate if concept drift has occurred.
        Requires results from multiple detectors to agree (consensus).
        """
        results = []
        
        # 1. Update ADWIN with mean confidence or similar
        # (Assuming errors or scores are passed)
        adwin_drift = any(self.adwin.update(e) for e in prediction_errors)
        results.append(adwin_drift)
        
        # 2. Update Page-Hinkley with error rate
        ph_drift = any(self.ph.update(e) for e in prediction_errors)
        results.append(ph_drift)
        
        # 3. Distributional tests on features
        ks_drift = self.ks.detect(current_batch)
        results.append(ks_drift)
        
        mmd_drift = self.mmd.detect(current_batch)
        results.append(mmd_drift)
        
        # Consensus: trigger if >= 2 agree
        return sum(results) >= 2
