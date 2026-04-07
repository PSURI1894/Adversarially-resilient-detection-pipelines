# Threat Model

## Adversary Capabilities

ARDP is designed against a Dolev-Yao-style adversary with partial system knowledge.

### Knowledge Assumptions

| Threat Level | Model Knowledge | Data Knowledge | Infrastructure Knowledge |
|---|---|---|---|
| White-box | Full gradient access | Full training set | Architecture known |
| Gray-box | Output probabilities only | Partial (surrogate training) | Unknown |
| Black-box | Hard labels only | None | Unknown |

### Perturbation Budgets

| Norm | Budget ε | Typical Scenario |
|---|---|---|
| ℓ∞ | 0.01 – 0.30 | Uniform bounded feature perturbation |
| ℓ₂ | 0.05 – 1.00 | Natural-looking traffic modification |

---

## Attack Taxonomy

### 1. White-Box Evasion Attacks

**PGD (ℓ∞ and ℓ₂)** — Projected Gradient Descent (Madry et al.)
- Iterative FGSM with projection onto ε-ball
- Random restarts to escape local minima
- Strongest gradient-based baseline

**Carlini & Wagner L2** — C&W L2 attack
- Binary search on constant c
- Optimised for tabular feature constraints (non-negative values)
- Often achieves lower distortion than PGD

**AutoAttack** — Ensemble of four sub-attacks:
- APGD-CE (cross-entropy loss)
- APGD-DLR (difference of logits ratio — label-leakage-free)
- FAB (Fast Adaptive Boundary)
- Square Attack (random search, black-box)
- Gold standard for robustness evaluation

### 2. Black-Box Evasion Attacks

**Boundary Attack** — Decision-based, rejection sampling + geometric steps
- Starts from adversarial side of boundary, walks towards target
- No gradient information required

**HopSkipJump Attack** — Gradient estimation via binary search on decision boundary
- Query-efficient; converges faster than Boundary Attack
- Realistic for APIs that return hard labels

**Transfer Attack** — Surrogate model + transferability
- Train surrogate on shadow data, craft examples, test transfer
- Models cross-model adversarial subspace overlap

### 3. Physical-Constraint Attacks

**Feature-Constrained Evasion**
- Only perturbs mutable flow features: inter-arrival time (IAT), packet length, duration
- Constraints: `bytes_sent ≥ 0`, temporal monotonicity, `IAT > 0`
- Respects TCP/IP stack invariants

**SlowDrip Attack**
- Stretches IAT features to mimic benign traffic burstiness patterns
- Low-and-slow exfiltration; designed to evade rate-based detectors

**Mimicry Attack**
- Maps malicious flow distributions to benign traffic clusters
- Uses Wasserstein barycenters to find minimal-distortion mapping
- Statistically indistinguishable from benign at flow level

### 4. Poisoning Attacks

**Label Flip Poisoning**
- Random or targeted flipping of training labels at contamination rate ρ ∈ [0, 0.30]
- Targeted variant maximises error on a specific attack category

**Backdoor Poisoning**
- Injects fixed trigger patterns (e.g., specific IAT spike) into N% of training flows
- Trigger activates `benign` classification at inference time
- Survives standard retraining unless trigger is detected

**Clean-Label Poisoning** (Shafahi et al.)
- Moves poison points to collide with target class boundary without label changes
- Bypasses label-consistency quality checks

**Calibration Poisoning**
- Specifically targets the conformal calibration set
- Inflates q_hat → larger prediction sets (coverage inflation attack)
- Deflates q_hat → under-coverage (silently violates guarantees)

### 5. Generative Adversarial Attacks

**WGAN-GP Adversary**
- Generator trained to produce flow records that fool the ensemble
- Feature constraints enforced via projection layer
- Discriminator doubles as auxiliary anomaly detector (adversarial co-training)

---

## Defence Assumptions

ARDP does **not** assume:
- The adversary's perturbation norm is known at inference time
- The calibration set is clean (calibration poisoning defence is included)
- The adversary cannot observe deployed model outputs (gray/black-box attacks are covered)

ARDP **does** assume:
- A clean, representative training set is available at initial training time
- At most ρ_max = 20% of calibration labels can be flipped simultaneously
- The adversary cannot break TLS to forge Kafka messages (network-layer security is out of scope)

---

## Risk Levels and Mitigations

| Risk Level | Trigger | Mitigation |
|---|---|---|
| LOW | Clean accuracy > 0.95, coverage > 0.95 | Standard operation |
| ELEVATED | Coverage drops 0.90–0.95 OR adversarial detector flags > 5% | Tighten alert threshold; log for review |
| HIGH | Coverage < 0.90 OR robust accuracy drop > 10% | Human-in-the-loop verification; reduce scope |
| CRISIS | Drift detected + coverage < 0.85 | Emergency retraining; fallback to rule-based IDS |
