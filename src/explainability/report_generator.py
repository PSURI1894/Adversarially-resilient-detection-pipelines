"""
================================================================================
INCIDENT REPORT GENERATOR — SOC-GRADE ALERT DOCUMENTATION
================================================================================
Generates structured JSON and HTML incident reports per alert, including:
    - Prediction + uncertainty set
    - SHAP waterfall data
    - LIME explanation
    - Risk score & SOC state
    - Severity-based prioritization
    - Analyst-actionable recommendations
================================================================================
"""

import json
import os
import datetime
import numpy as np
from typing import Optional, List, Dict, Any


class IncidentReporter:
    """
    Generates structured incident reports for SOC analysts.

    Parameters
    ----------
    output_dir : str
        Directory to write reports to.
    feature_names : list of str, optional
        Human-readable feature names.
    """

    def __init__(
        self,
        output_dir: str = "reports/incidents",
        feature_names: Optional[List[str]] = None,
    ):
        self.output_dir = output_dir
        self.feature_names = feature_names
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Single incident report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        sample: np.ndarray,
        prediction: float,
        prediction_set: List[int],
        shap_explanation: Optional[Dict] = None,
        lime_explanation: Optional[Dict] = None,
        risk_score: float = 0.0,
        soc_state: str = "STABLE",
        alert_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a structured incident report.

        Returns
        -------
        dict: The complete report as a structured dictionary.
        """
        alert_id = (
            alert_id or f"ALERT-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        )

        severity = self._compute_severity(
            prediction, len(prediction_set), risk_score, soc_state
        )
        priority = self._severity_to_priority(severity)

        report = {
            "alert_id": alert_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "severity": severity,
            "priority": priority,
            "prediction": {
                "malicious_probability": float(prediction),
                "label": "MALICIOUS" if prediction > 0.5 else "BENIGN",
                "prediction_set": prediction_set,
                "uncertainty": "HIGH" if len(prediction_set) > 1 else "LOW",
            },
            "risk": {
                "score": float(risk_score),
                "soc_state": soc_state,
            },
            "explanations": {
                "shap": self._format_shap(shap_explanation)
                if shap_explanation
                else None,
                "lime": self._format_lime(lime_explanation)
                if lime_explanation
                else None,
            },
            "recommendations": self._generate_recommendations(
                prediction,
                prediction_set,
                risk_score,
                soc_state,
                shap_explanation,
                lime_explanation,
            ),
            "sample_features": self._format_sample(sample),
        }

        return report

    # ------------------------------------------------------------------
    # Batch reporting
    # ------------------------------------------------------------------

    def generate_batch_reports(
        self,
        X: np.ndarray,
        predictions: np.ndarray,
        prediction_sets: List[List[int]],
        risk_scores: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Generate reports for an entire batch."""
        reports = []
        for i in range(len(X)):
            report = self.generate_report(
                sample=X[i],
                prediction=float(predictions[i]),
                prediction_set=prediction_sets[i],
                risk_score=float(risk_scores[i]) if risk_scores is not None else 0.0,
            )
            reports.append(report)
        return reports

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_json(
        self, report: Dict[str, Any], filename: Optional[str] = None
    ) -> str:
        """Export a single report to JSON file."""
        filename = filename or f"{report['alert_id']}.json"
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        return path

    def export_html(
        self, report: Dict[str, Any], filename: Optional[str] = None
    ) -> str:
        """Export a single report as an HTML page."""
        filename = filename or f"{report['alert_id']}.html"
        path = os.path.join(self.output_dir, filename)

        html = self._render_html(report)
        with open(path, "w") as f:
            f.write(html)
        return path

    def export_csv_summary(
        self, reports: List[Dict[str, Any]], filename: str = "summary.csv"
    ) -> str:
        """Export batch summary to CSV."""
        path = os.path.join(self.output_dir, filename)
        lines = [
            "alert_id,timestamp,severity,priority,prediction,uncertainty,risk_score,soc_state"
        ]
        for r in reports:
            lines.append(
                f"{r['alert_id']},{r['timestamp']},{r['severity']},{r['priority']},"
                f"{r['prediction']['malicious_probability']:.4f},"
                f"{r['prediction']['uncertainty']},{r['risk']['score']:.2f},"
                f"{r['risk']['soc_state']}"
            )
        with open(path, "w") as f:
            f.write("\n".join(lines))
        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_severity(self, prediction, set_size, risk_score, soc_state):
        """Continuous severity 0-100."""
        pred_component = prediction * 40
        uncertainty_component = (set_size - 1) * 20
        risk_component = risk_score * 0.3
        state_bonus = {"FAILURE": 20, "EVASION_LOCKED": 15, "SUSPICIOUS": 5}.get(
            soc_state, 0
        )
        return min(
            100, pred_component + uncertainty_component + risk_component + state_bonus
        )

    def _severity_to_priority(self, severity):
        if severity >= 80:
            return "P1-CRITICAL"
        elif severity >= 50:
            return "P2-HIGH"
        elif severity >= 25:
            return "P3-MEDIUM"
        return "P4-LOW"

    def _format_shap(self, explanation):
        return {
            "base_value": explanation.get("base_value"),
            "top_features": explanation.get("top_features", []),
        }

    def _format_lime(self, explanation):
        return {
            "fidelity": explanation.get("fidelity"),
            "top_features": explanation.get("top_features", []),
        }

    def _format_sample(self, sample):
        sample = np.atleast_1d(sample).flatten()
        names = self.feature_names or [f"f{i}" for i in range(len(sample))]
        return dict(zip(names, [float(v) for v in sample]))

    def _generate_recommendations(
        self, prediction, prediction_set, risk_score, soc_state, shap_exp, lime_exp
    ):
        recs = []
        if prediction > 0.8:
            recs.append("BLOCK: Immediately quarantine source IP and notify Tier-2.")
        elif prediction > 0.5:
            recs.append(
                "INVESTIGATE: Manual analyst review recommended within 15 minutes."
            )

        if len(prediction_set) > 1:
            recs.append("UNCERTAINTY: Model is unsure — escalate to senior analyst.")

        if soc_state in ("FAILURE", "EVASION_LOCKED"):
            recs.append(
                "SYSTEM ALERT: SOC is under stress — activate incident response playbook."
            )

        if shap_exp and shap_exp.get("top_features"):
            top_f = shap_exp["top_features"][0][0]
            recs.append(f"KEY DRIVER: '{top_f}' is the primary attribution signal.")

        if not recs:
            recs.append("ROUTINE: No immediate action required.")

        return recs

    def _render_html(self, report):
        severity_color = {
            "P1-CRITICAL": "#dc3545",
            "P2-HIGH": "#fd7e14",
            "P3-MEDIUM": "#ffc107",
            "P4-LOW": "#28a745",
        }.get(report["priority"], "#6c757d")

        recs_html = "".join(f"<li>{r}</li>" for r in report["recommendations"])
        shap_html = ""
        if report["explanations"]["shap"]:
            rows = "".join(
                f"<tr><td>{name}</td><td>{val:.4f}</td></tr>"
                for name, val in report["explanations"]["shap"]["top_features"][:5]
            )
            shap_html = f"<h3>SHAP Attribution</h3><table><tr><th>Feature</th><th>Value</th></tr>{rows}</table>"

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Incident Report — {report["alert_id"]}</title>
    <style>
        body {{ font-family: 'Inter', sans-serif; background: #0d1117; color: #c9d1d9; padding: 2rem; }}
        .card {{ background: #161b22; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; border: 1px solid #30363d; }}
        .priority {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-weight: 700; color: white; background: {severity_color}; }}
        h1 {{ color: #58a6ff; }} h2,h3 {{ color: #79c0ff; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th,td {{ padding: 8px 12px; border-bottom: 1px solid #30363d; text-align: left; }}
        th {{ color: #8b949e; }}
        li {{ margin: 4px 0; }}
    </style>
</head>
<body>
    <h1>🛡️ Incident Report</h1>
    <div class="card">
        <strong>Alert ID:</strong> {report["alert_id"]}<br>
        <strong>Timestamp:</strong> {report["timestamp"]}<br>
        <strong>Severity:</strong> {report["severity"]:.1f} <span class="priority">{report["priority"]}</span>
    </div>
    <div class="card">
        <h2>Prediction</h2>
        <strong>Label:</strong> {report["prediction"]["label"]}<br>
        <strong>P(malicious):</strong> {report["prediction"]["malicious_probability"]:.4f}<br>
        <strong>Prediction Set:</strong> {report["prediction"]["prediction_set"]}<br>
        <strong>Uncertainty:</strong> {report["prediction"]["uncertainty"]}
    </div>
    <div class="card">
        <h2>Risk Assessment</h2>
        <strong>Score:</strong> {report["risk"]["score"]:.2f}<br>
        <strong>SOC State:</strong> {report["risk"]["soc_state"]}
    </div>
    <div class="card">{shap_html}</div>
    <div class="card">
        <h2>Recommendations</h2>
        <ul>{recs_html}</ul>
    </div>
</body>
</html>"""
