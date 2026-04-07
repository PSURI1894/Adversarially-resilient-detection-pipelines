# Attack Catalog

Complete documentation of all adversarial attack implementations in `src/attacks/`.

---

## White-Box Attacks (`src/attacks/white_box.py`)

### `PGDAttack`

Projected Gradient Descent (Madry et al., 2018).

**Parameters**

| Param | Default | Description |
|---|---|---|
| `epsilon` | 0.1 | Perturbation budget |
| `norm` | `"l_inf"` | `"l_inf"` or `"l_2"` |
| `max_iter` | 20 | Number of PGD steps |
| `step_size` | auto | Step size (default: 2.5ε/max_iter) |
| `random_start` | True | Random initialisation in ε-ball |

**Usage**
```python
from src.attacks.white_box import PGDAttack, AttackConfig

cfg = AttackConfig(epsilon=0.1, norm="l_inf", max_iter=20)
attack = PGDAttack(cfg)
X_adv = attack.generate(model, X_test, y_test)
```

**Threat level**: HIGH — requires gradient access but is the de facto standard
robustness evaluation.

---

### `CarliniWagnerL2`

C&W L2 attack (Carlini & Wagner, 2017).

**Parameters**

| Param | Default | Description |
|---|---|---|
| `epsilon` | 0.1 | Initial c constant (binary-searched) |
| `max_iter` | 100 | Optimisation iterations |
| `confidence` | 0 | κ — logit gap to enforce |

**Notes**: Typically achieves lower distortion than PGD but is slower (100× iterations).
Best suited for evaluation, not real-time attack simulation.

---

### `AutoAttack`

Ensemble of four sub-attacks (Croce & Hein, 2020). Gold standard for robustness evaluation.

Sub-attacks: APGD-CE, APGD-DLR, FAB, Square Attack.

**Parameters**

| Param | Default | Description |
|---|---|---|
| `epsilon` | 0.1 | Shared perturbation budget |
| `norm` | `"l_inf"` | `"l_inf"` or `"l_2"` |
| `version` | `"standard"` | `"standard"` or `"rand"` (randomised) |

---

## Black-Box Attacks (`src/attacks/black_box.py`)

### `BoundaryAttack`

Decision-based attack (Brendel et al., 2018).

Starts from an adversarial image and walks towards the target while staying adversarial.
Requires only binary (hard-label) feedback.

**Parameters**

| Param | Default | Description |
|---|---|---|
| `max_iter` | 5000 | Walk steps |
| `spherical_step` | 0.01 | Angular step size |
| `source_step` | 0.01 | Source direction step |

---

### `HopSkipJumpAttack`

Query-efficient decision-based attack (Chen et al., 2020).

Uses binary search on the decision boundary to estimate gradients.
Achieves convergence ~10× faster than Boundary Attack for tabular data.

---

### `TransferAttack`

1. Trains a surrogate model (logistic regression / small MLP) on shadow data
2. Crafts adversarial examples against the surrogate using PGD
3. Evaluates transferability to the target ensemble

**Transferability rate** on CIC-IDS2018: ~60% for ℓ∞ examples, ~45% for ℓ₂.

---

## Physical-Constraint Attacks (`src/attacks/physical.py`)

### `FeatureConstrainedEvasion`

Only perturbs **mutable** flow features. Enforces:
- `bytes_sent ≥ 0`, `bytes_recv ≥ 0`
- `duration ≥ 0`, temporal monotonicity
- `IAT > 0` (inter-arrival time strictly positive)

Mutable features (CIC-IDS2018 feature indices): `[2, 4, 5, 7, 14, 23, 31, 48]`
(IAT mean/std, packet length, flow duration, forward/backward stats).

---

### `SlowDripAttack`

Low-and-slow exfiltration that stretches IAT features to mimic benign burstiness.

**Effect on IDS**: rate-based detectors lose signal; connection must be long-lived.
**Countermeasure in ARDP**: temporal feature store captures 60-second windows.

---

### `MimicryAttack`

Maps malicious flow distribution to the nearest benign cluster using Wasserstein barycenters.
Statistically indistinguishable at the flow-feature level; detectable only via temporal correlation.

---

## Poisoning Attacks (`src/attacks/poisoning.py`)

### `LabelFlipPoisoning`

| Variant | Effect |
|---|---|
| `random` | Flips `ρ` fraction of labels uniformly |
| `targeted` | Maximises error on a specific attack category |

**Detection**: compare label distribution before/after; monitor q_hat drift.

---

### `BackdoorPoisoning`

- Trigger: fixed IAT spike pattern (e.g., 4× normal IAT at packet 5)
- Effect: trigger at inference → predicted benign regardless of actual attack
- ARDP defence: SHAP fingerprint anomaly score detects atypical attribution pattern for triggered flows

---

### `CleanLabelPoisoning`

Moves poison points toward the target class decision boundary without label changes.
Bypasses label-consistency checks. Requires white-box access to craft.

---

### `CalibrationPoisoning`

Directly attacks the conformal prediction calibration set:
- **Inflation attack**: push q_hat up → larger sets → alert fatigue
- **Deflation attack**: push q_hat down → coverage drops silently

**ARDP defence**: `src/conformal/poison_defense.py` — robust quantile estimation
using weighted trimmed mean to discard outlier calibration scores.

---

## GAN-Based Adversary (`src/attacks/gan_adversary.py`)

### `AdversarialGAN`

Wasserstein GAN with gradient penalty (WGAN-GP).

- **Generator**: Dense network; learns to produce traffic that fools the ensemble
- **Discriminator**: Dense network; doubles as auxiliary anomaly detector
- **Training objective**: generator minimises Wasserstein distance from decision boundary
- **Feature constraints**: projection layer enforces non-negativity + flow invariants

**Generator architecture**: `z(100) → Dense(256) → Dense(128) → Dense(d) → Clip(0, ∞)`

**Usage**
```python
from src.attacks.gan_adversary import AdversarialGAN

gan = AdversarialGAN(input_dim=80, latent_dim=100)
gan.train(X_benign, X_malicious, epochs=200)
X_adv = gan.generate(n_samples=1000)
```

**Discriminator as detector**: after training, `gan.discriminator.score(X)` returns
a score in [0,1] (1 = adversarial). Can be used as a standalone detector.

---

## Attack Registry

All attacks are registered in `src/attacks/__init__.py` and can be instantiated
by name for sweep experiments:

```python
from src.attacks import get_attack

attack = get_attack("pgd_linf", epsilon=0.1)
X_adv = attack.generate(model, X_test, y_test)
```

Registry keys: `pgd_linf`, `pgd_l2`, `carlini_wagner`, `autoattack`,
`boundary`, `hopskipjump`, `transfer`, `physical`, `slow_drip`, `mimicry`,
`label_flip`, `backdoor`, `clean_label`, `calibration_poison`, `gan`.
