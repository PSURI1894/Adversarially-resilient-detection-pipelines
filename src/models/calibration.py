"""
Calibration module for Deep Ensembles and other detection models.
Temperature Scaling, Isotonic Regression, and Calibration Auditing.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss

class TemperatureScaling:
    """Post-hoc Platt scaling with a single learned temperature parameter."""
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits_val, y_val):
        """Fit the temperature parameter using NLL."""
        # Objective: minimize NLL
        def objective(tau):
            t = tau[0]
            # scale logits
            scaled_logits = logits_val / t
            # compute sigmoid loss
            loss = np.mean(np.maximum(scaled_logits, 0) - scaled_logits * y_val + np.log1p(np.exp(-np.abs(scaled_logits))))
            return loss

        # Initial guess 1.0
        bounds = [(0.01, 100.0)]
        result = minimize(objective, [1.0], bounds=bounds, method='L-BFGS-B')
        self.temperature = result.x[0]

    def predict_proba(self, logits):
        """Returns calibrated probabilities from logits."""
        scaled = logits / self.temperature
        probs = 1 / (1 + np.exp(-scaled))
        return np.vstack([1 - probs, probs]).T


class IsotonicCalibration:
    """Non-parametric monotone calibration using Isotonic Regression."""
    def __init__(self):
        self.ir = IsotonicRegression(out_of_bounds='clip')

    def fit(self, probs_val, y_val):
        """Fit strictly on internal probability for positive class."""
        if len(probs_val.shape) == 2:
            probs_val = probs_val[:, 1]
        self.ir.fit(probs_val, y_val)

    def predict_proba(self, probs):
        if len(probs.shape) == 2:
            p = probs[:, 1]
        else:
            p = probs
        calibrated_p = self.ir.transform(p)
        return np.vstack([1 - calibrated_p, calibrated_p]).T


class CalibrationAudit:
    """Compute Expected Calibration Error (ECE) and other metrics."""
    
    @staticmethod
    def expected_calibration_error(y_true, y_prob, n_bins=10):
        if len(y_prob.shape) == 2:
            y_prob = y_prob[:, 1]
            
        bins = np.linspace(0, 1, n_bins + 1)
        binids = np.digitize(y_prob, bins) - 1
        
        ece = 0.0
        for i in range(n_bins):
            bin_mask = binids == i
            if np.sum(bin_mask) > 0:
                bin_acc = np.mean(y_true[bin_mask])
                bin_conf = np.mean(y_prob[bin_mask])
                ece += np.abs(bin_acc - bin_conf) * np.sum(bin_mask) / len(y_true)
        return ece

    @staticmethod
    def maximum_calibration_error(y_true, y_prob, n_bins=10):
        if len(y_prob.shape) == 2:
            y_prob = y_prob[:, 1]
            
        bins = np.linspace(0, 1, n_bins + 1)
        binids = np.digitize(y_prob, bins) - 1
        
        mce = 0.0
        for i in range(n_bins):
            bin_mask = binids == i
            if np.sum(bin_mask) > 0:
                bin_acc = np.mean(y_true[bin_mask])
                bin_conf = np.mean(y_prob[bin_mask])
                mce = max(mce, np.abs(bin_acc - bin_conf))
        return mce

    @staticmethod
    def reliability_diagram_data(y_true, y_prob, n_bins=10):
        if len(y_prob.shape) == 2:
            y_prob = y_prob[:, 1]
            
        bins = np.linspace(0, 1, n_bins + 1)
        binids = np.digitize(y_prob, bins) - 1
        
        bin_accs = []
        bin_confs = []
        for i in range(n_bins):
            bin_mask = binids == i
            if np.sum(bin_mask) > 0:
                bin_accs.append(np.mean(y_true[bin_mask]))
                bin_confs.append(np.mean(y_prob[bin_mask]))
            else:
                bin_accs.append(np.nan)
                bin_confs.append(np.nan)
                
        return np.array(bin_confs), np.array(bin_accs)
