# Contributing Guide

Thank you for your interest in ARDP. This guide explains how to contribute
code, documentation, or attack/defense implementations.

---

## Development Setup

```bash
git clone <repo-url>
cd Adversarially-resilient-detection-pipelines
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-cov black isort mypy
```

---

## Code Style

- **Formatter**: `black` (line length 100)
- **Import sorter**: `isort`
- **Type hints**: required on all public functions and class methods
- **Docstrings**: Google-style for new modules/classes; keep existing style if extending

Run before committing:
```bash
black src/ tests/ experiments/ --line-length 100
isort src/ tests/ experiments/
```

---

## Testing

All new code must include tests. Coverage target: **≥ 90%** on `src/`.

```bash
# Run all tests
pytest tests/ -v --tb=short

# Coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run a specific file
pytest tests/test_attacks.py -v
```

### Test conventions

- Unit tests go in `tests/test_<module>.py`
- Integration tests go in `tests/test_integration.py`
- Use `numpy.random.seed(42)` in all test fixtures for reproducibility
- Tests must pass without GPU; use synthetic data or small CPU-friendly fixtures

---

## Adding a New Attack

1. Add your class to the appropriate file in `src/attacks/` (or create a new file)
2. Inherit from `BaseAttack` and implement `generate(model, X, y) -> np.ndarray`
3. Register in `src/attacks/__init__.py`
4. Add ≥ 3 test cases in `tests/test_attacks.py`:
   - ε-budget compliance test
   - Feature constraint test (for physical attacks)
   - Output shape test
5. Document in `docs/attack_catalog.md`

```python
# src/attacks/my_attack.py
from src.attacks.white_box import BaseAttack, AttackConfig
import numpy as np

class MyAttack(BaseAttack):
    """One-line description."""

    def generate(self, model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # ... implementation
        return X_adv
```

---

## Adding a New Defense

1. Add to the appropriate `src/` subpackage
2. If it modifies conformal behavior: add integration test in `tests/test_integration.py`
3. If it affects the Risk FSM: update `src/risk_management_engine.py` and document state transitions
4. Update `docs/architecture.md` with the new component

---

## Pull Request Checklist

- [ ] Tests pass locally (`pytest tests/ -v`)
- [ ] Coverage does not drop below 90% on changed files
- [ ] `black` + `isort` applied
- [ ] Type hints added/updated
- [ ] `docs/` updated if public API or behavior changed
- [ ] PR description explains *why* the change is needed (not just what)

---

## Branching Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, reviewed code |
| `week-N` | Weekly feature branches (merged via PR) |
| `fix/<short-description>` | Bug fixes |
| `exp/<name>` | Experimental branches (not merged to main) |

---

## Reporting Issues

Use GitHub Issues with the following labels:

- `bug` — reproducible defect with steps to reproduce
- `enhancement` — new feature or improvement
- `attack` — new attack implementation
- `defense` — new defense mechanism
- `docs` — documentation improvement
- `performance` — latency / memory issue

Include: Python version, OS, steps to reproduce, expected vs actual behavior.
