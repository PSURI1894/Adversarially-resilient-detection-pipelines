"""
================================================================================
ADVERSARIAL ATTACK LIBRARY
================================================================================
Research-grade attack arsenal for evaluating IDS robustness.
Covers white-box, black-box, physical-constraint, poisoning, and GAN attacks.
================================================================================
"""

from src.attacks.white_box import PGDAttack, CarliniWagnerL2, AutoAttack
from src.attacks.black_box import BoundaryAttack, HopSkipJumpAttack, TransferAttack
from src.attacks.physical import (
    FeatureConstrainedEvasion,
    SlowDripAttack,
    MimicryAttack,
)
from src.attacks.poisoning import (
    LabelFlipPoisoning,
    BackdoorPoisoning,
    CleanLabelPoisoning,
    CalibrationPoisoning,
)
from src.attacks.gan_adversary import AdversarialGAN

# ── registry ────────────────────────────────────────────────────
ATTACK_REGISTRY = {
    # White-box
    "pgd": PGDAttack,
    "cw_l2": CarliniWagnerL2,
    "auto": AutoAttack,
    # Black-box
    "boundary": BoundaryAttack,
    "hsja": HopSkipJumpAttack,
    "transfer": TransferAttack,
    # Physical
    "feature_constrained": FeatureConstrainedEvasion,
    "slow_drip": SlowDripAttack,
    "mimicry": MimicryAttack,
    # Poisoning
    "label_flip": LabelFlipPoisoning,
    "backdoor": BackdoorPoisoning,
    "clean_label": CleanLabelPoisoning,
    "calibration_poison": CalibrationPoisoning,
    # Generative
    "gan": AdversarialGAN,
}

__all__ = list(ATTACK_REGISTRY.keys()) + ["ATTACK_REGISTRY"]
