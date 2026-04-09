"""
================================================================================
AWS Lambda — ARDP Health Check Function
================================================================================
Invoked every 5 minutes by EventBridge.
Checks:
  1. /api/status endpoint reachability
  2. Model coverage within acceptable bounds
  3. Drift detector state
  4. Publishes CloudWatch metrics for alerting

Environment variables:
  ARDP_API_URL     — base URL of the FastAPI backend (e.g. http://1.2.3.4:8000)
  COVERAGE_MIN     — minimum acceptable conformal coverage (default 0.85)
  SNS_TOPIC_ARN    — (optional) SNS topic ARN for alert notifications
================================================================================
"""

import json
import os
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Any, Dict

import boto3

# ── Configuration ─────────────────────────────────────────────
API_URL = os.environ.get("ARDP_API_URL", "http://localhost:8000").rstrip("/")
COVERAGE_MIN = float(os.environ.get("COVERAGE_MIN", "0.85"))
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
TIMEOUT_SEC = 10

cloudwatch = boto3.client("cloudwatch")
sns = boto3.client("sns") if SNS_TOPIC_ARN else None


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _get(path: str) -> Dict[str, Any]:
    url = f"{API_URL}{path}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
        return json.loads(resp.read().decode())


def _put_metric(name: str, value: float, unit: str = "None") -> None:
    cloudwatch.put_metric_data(
        Namespace="ARDP/Pipeline",
        MetricData=[{
            "MetricName": name,
            "Value": value,
            "Unit": unit,
            "Timestamp": datetime.now(tz=timezone.utc),
            "Dimensions": [{"Name": "Environment", "Value": "Production"}],
        }],
    )


def _alert(subject: str, message: str) -> None:
    if sns and SNS_TOPIC_ARN:
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=f"[ARDP Alert] {subject}",
            Message=message,
        )


# ─────────────────────────────────────────────────────────────
# Handler
# ─────────────────────────────────────────────────────────────

def handler(event: Dict, context: Any) -> Dict[str, Any]:
    issues = []
    metrics = {}

    # ── 1. API reachability ───────────────────────────────────
    try:
        status = _get("/api/status")
        metrics["api_reachable"] = 1.0
        metrics["soc_severity"] = float(status.get("severity", 0))
        metrics["alert_debt"] = float(status.get("alert_debt", 0))
        soc_state = status.get("soc_state", "UNKNOWN")
        metrics["soc_state_numeric"] = {"STABLE": 0, "ELEVATED": 1, "CRISIS": 2}.get(soc_state, 3)
    except Exception as exc:
        metrics["api_reachable"] = 0.0
        issues.append(f"API unreachable: {exc}")

    # ── 2. Conformal coverage ─────────────────────────────────
    try:
        m = _get("/api/metrics")
        # Coverage is reported inside the metrics history or status
        coverage = float(
            m.get("conformal_coverage_last_batch")
            or m.get("coverage_1000")
            or 1.0
        )
        metrics["conformal_coverage"] = coverage
        if coverage < COVERAGE_MIN:
            issues.append(
                f"Conformal coverage {coverage:.3f} below minimum {COVERAGE_MIN:.3f}"
            )
    except Exception:
        pass  # non-fatal; status check already covers API availability

    # ── 3. Drift state ────────────────────────────────────────
    try:
        drift = _get("/api/drift/status")
        consensus = drift.get("consensus_state", "UNKNOWN")
        metrics["drift_elevated"] = 1.0 if consensus != "NORMAL" else 0.0
        if consensus == "CRISIS":
            issues.append(f"Drift state is CRISIS — retraining may be needed")
    except Exception:
        pass

    # ── 4. Publish CloudWatch metrics ─────────────────────────
    for name, value in metrics.items():
        try:
            _put_metric(name, value)
        except Exception:
            pass

    # ── 5. Send alert if issues found ─────────────────────────
    healthy = len(issues) == 0
    if not healthy:
        msg = "\n".join(f"• {i}" for i in issues)
        full_msg = (
            f"ARDP health check failed at {datetime.now(tz=timezone.utc).isoformat()}\n\n"
            f"Issues detected:\n{msg}\n\n"
            f"API: {API_URL}\n"
            f"Metrics: {json.dumps(metrics, indent=2)}"
        )
        _alert("Health Check Failed", full_msg)
        print(f"[ALERT] {full_msg}")
    else:
        print(f"[OK] ARDP healthy — metrics: {metrics}")

    return {
        "statusCode": 200,
        "healthy": healthy,
        "issues": issues,
        "metrics": metrics,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
