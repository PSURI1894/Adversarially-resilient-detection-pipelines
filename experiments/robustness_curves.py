"""
================================================================================
ROBUSTNESS CURVES
================================================================================
Publication-quality plots (IEEE/Nature format) for:
  1. ε (attack strength) vs. Robust Accuracy under PGD, C&W, AutoAttack
  2. ε vs. Conformal Coverage with RSCP+ certified bound overlay
  3. Poisoning fraction vs. Calibration Validity (q_hat stability)
  4. Ensemble size vs. Epistemic Uncertainty

All figures saved to paper/figures/ and experiments/results/figures/.
================================================================================
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

FIGURES_DIR = Path(__file__).parent.parent / "paper" / "figures"
RESULTS_FIGURES_DIR = Path(__file__).parent / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# IEEE single-column figure size
IEEE_FIGSIZE = (3.5, 2.625)      # inches (single column)
IEEE_FIGSIZE_WIDE = (7.16, 2.8)  # double column


def _apply_ieee_style(ax, legend: bool = True) -> None:
    """Apply publication-quality formatting to a matplotlib Axes."""
    import matplotlib.pyplot as plt
    ax.tick_params(axis="both", labelsize=8, direction="in", top=True, right=True)
    ax.set_xlabel(ax.get_xlabel(), fontsize=9)
    ax.set_ylabel(ax.get_ylabel(), fontsize=9)
    if legend:
        ax.legend(fontsize=7, framealpha=0.8, edgecolor="0.8")
    ax.spines["top"].set_linewidth(0.5)
    ax.spines["right"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — ε vs Robust Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def plot_epsilon_vs_accuracy(
    results_by_attack: Dict[str, Dict[str, List]],
    save: bool = True,
) -> None:
    """
    Parameters
    ----------
    results_by_attack : dict
        {attack_name: {"epsilons": [...], "robust_acc": [...], "clean_acc": float}}
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=IEEE_FIGSIZE)

    colors = {"pgd_linf": "#e41a1c", "pgd_l2": "#377eb8",
              "carlini_wagner": "#4daf4a", "autoattack": "#984ea3",
              "boundary": "#ff7f00"}
    markers = {"pgd_linf": "o", "pgd_l2": "s", "carlini_wagner": "^",
               "autoattack": "D", "boundary": "v"}
    labels = {"pgd_linf": "PGD ($\\ell_\\infty$)", "pgd_l2": "PGD ($\\ell_2$)",
              "carlini_wagner": "C\\&W $\\ell_2$", "autoattack": "AutoAttack",
              "boundary": "Boundary"}

    # Draw clean accuracy reference line (assume same for all attacks)
    first = next(iter(results_by_attack.values()))
    if "clean_acc" in first:
        ax.axhline(first["clean_acc"], color="black", linestyle="--",
                   linewidth=0.8, label="Clean acc.", zorder=5)

    for attack, data in results_by_attack.items():
        eps = data["epsilons"]
        rob = data["robust_acc"]
        c = colors.get(attack, "gray")
        m = markers.get(attack, "o")
        lbl = labels.get(attack, attack)
        ax.plot(eps, rob, marker=m, markersize=4, linewidth=1.2, color=c, label=lbl)

    ax.set_xlabel("Perturbation budget $\\varepsilon$")
    ax.set_ylabel("Robust accuracy")
    ax.set_ylim(0, 1.05)
    _apply_ieee_style(ax)

    fig.tight_layout(pad=0.3)
    if save:
        for d in (FIGURES_DIR, RESULTS_FIGURES_DIR):
            fig.savefig(d / "epsilon_vs_accuracy.pdf", dpi=300, bbox_inches="tight")
            fig.savefig(d / "epsilon_vs_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: epsilon_vs_accuracy.[pdf|png]")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — ε vs Conformal Coverage + certified bound
# ─────────────────────────────────────────────────────────────────────────────

def plot_epsilon_vs_coverage(
    epsilons: List[float],
    empirical_coverage: List[float],
    certified_bound: Optional[List[float]] = None,
    set_sizes: Optional[List[float]] = None,
    alpha: float = 0.05,
    save: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2 if set_sizes else 1,
                             figsize=(IEEE_FIGSIZE_WIDE if set_sizes else IEEE_FIGSIZE))
    ax_cov = axes[0] if set_sizes else axes

    ax_cov.plot(epsilons, empirical_coverage, "b-o", markersize=4,
                linewidth=1.2, label="Empirical coverage")
    ax_cov.axhline(1 - alpha, color="black", linestyle="--", linewidth=0.8,
                   label=f"Target ($1-\\alpha={1-alpha:.2f}$)")

    if certified_bound is not None:
        ax_cov.fill_between(epsilons, certified_bound, [1.0] * len(epsilons),
                            alpha=0.15, color="green", label="RSCP+ certified region")
        ax_cov.plot(epsilons, certified_bound, "g--", linewidth=0.9,
                    label="Certified lower bound")

    ax_cov.set_xlabel("Perturbation budget $\\varepsilon$")
    ax_cov.set_ylabel("Conformal coverage")
    ax_cov.set_ylim(0.5, 1.05)
    _apply_ieee_style(ax_cov)

    if set_sizes:
        ax_sz = axes[1]
        ax_sz.plot(epsilons, set_sizes, "r-s", markersize=4, linewidth=1.2,
                   label="Avg. set size")
        ax_sz.set_xlabel("Perturbation budget $\\varepsilon$")
        ax_sz.set_ylabel("Average prediction-set size")
        _apply_ieee_style(ax_sz)

    fig.tight_layout(pad=0.3)
    if save:
        for d in (FIGURES_DIR, RESULTS_FIGURES_DIR):
            fig.savefig(d / "epsilon_vs_coverage.pdf", dpi=300, bbox_inches="tight")
            fig.savefig(d / "epsilon_vs_coverage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: epsilon_vs_coverage.[pdf|png]")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Poisoning fraction vs Calibration Validity
# ─────────────────────────────────────────────────────────────────────────────

def plot_poisoning_vs_calibration(
    poison_fractions: List[float],
    q_hat_values: List[float],
    baseline_q_hat: float,
    coverage_values: Optional[List[float]] = None,
    save: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=IEEE_FIGSIZE)
    ax.plot(poison_fractions, q_hat_values, "m-o", markersize=4,
            linewidth=1.2, label="$\\hat{q}$ (poisoned)")
    ax.axhline(baseline_q_hat, color="black", linestyle="--", linewidth=0.8,
               label="$\\hat{q}$ (clean baseline)")

    if coverage_values:
        ax2 = ax.twinx()
        ax2.plot(poison_fractions, coverage_values, "c-^", markersize=4,
                 linewidth=1.0, alpha=0.7, label="CP coverage")
        ax2.set_ylabel("Conformal coverage", fontsize=9)
        ax2.tick_params(labelsize=8)

    ax.set_xlabel("Poisoning fraction")
    ax.set_ylabel("Conformal threshold $\\hat{q}$")
    _apply_ieee_style(ax, legend=True)
    ax.set_title("Calibration Validity under Label Poisoning", fontsize=9)

    fig.tight_layout(pad=0.3)
    if save:
        for d in (FIGURES_DIR, RESULTS_FIGURES_DIR):
            fig.savefig(d / "poisoning_vs_calibration.pdf", dpi=300, bbox_inches="tight")
            fig.savefig(d / "poisoning_vs_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: poisoning_vs_calibration.[pdf|png]")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Ensemble size vs Epistemic Uncertainty
# ─────────────────────────────────────────────────────────────────────────────

def plot_ensemble_uncertainty(
    ensemble_sizes: List[int],
    mean_uncertainties: List[float],
    std_uncertainties: Optional[List[float]] = None,
    save: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=IEEE_FIGSIZE)
    ax.plot(ensemble_sizes, mean_uncertainties, "b-o", markersize=4, linewidth=1.2)
    if std_uncertainties:
        lo = [m - s for m, s in zip(mean_uncertainties, std_uncertainties)]
        hi = [m + s for m, s in zip(mean_uncertainties, std_uncertainties)]
        ax.fill_between(ensemble_sizes, lo, hi, alpha=0.2, color="blue")

    ax.set_xlabel("Number of ensemble members")
    ax.set_ylabel("Mean epistemic uncertainty")
    ax.set_title("Epistemic Uncertainty vs. Ensemble Size", fontsize=9)
    _apply_ieee_style(ax, legend=False)

    fig.tight_layout(pad=0.3)
    if save:
        for d in (FIGURES_DIR, RESULTS_FIGURES_DIR):
            fig.savefig(d / "ensemble_uncertainty.pdf", dpi=300, bbox_inches="tight")
            fig.savefig(d / "ensemble_uncertainty.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: ensemble_uncertainty.[pdf|png]")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: generate all plots from saved benchmark JSON
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_from_results(json_path: str | Path) -> None:
    """Load a benchmark_suite JSON and regenerate all robustness curve plots."""
    with open(json_path) as f:
        records = json.load(f)

    # Organise by attack name
    by_attack: Dict[str, Dict] = {}
    for r in records:
        name = r["attack_name"]
        if name not in by_attack:
            by_attack[name] = {"epsilons": [], "robust_acc": [], "clean_acc": r["clean_accuracy"]}
        by_attack[name]["epsilons"].append(r["epsilon"])
        by_attack[name]["robust_acc"].append(r["robust_accuracy"])

    plot_epsilon_vs_accuracy(by_attack)

    # Coverage curves (aggregated across all attacks for first attack found)
    first_attack = next(iter(by_attack))
    eps = by_attack[first_attack]["epsilons"]
    cov = [r["conformal_coverage"] for r in records if r["attack_name"] == first_attack]
    sizes = [r["avg_set_size"] for r in records if r["attack_name"] == first_attack]
    if len(cov) == len(eps):
        plot_epsilon_vs_coverage(eps, cov, set_sizes=sizes)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic demo
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_demo() -> None:
    """Generate all plots with synthetic data for smoke-testing."""
    eps = [0.01, 0.05, 0.1, 0.2, 0.3]

    results_by_attack = {
        "pgd_linf": {
            "epsilons": eps,
            "robust_acc": [0.95, 0.88, 0.78, 0.63, 0.50],
            "clean_acc": 0.97,
        },
        "pgd_l2": {
            "epsilons": eps,
            "robust_acc": [0.96, 0.91, 0.83, 0.71, 0.58],
            "clean_acc": 0.97,
        },
        "carlini_wagner": {
            "epsilons": eps,
            "robust_acc": [0.93, 0.82, 0.70, 0.55, 0.42],
            "clean_acc": 0.97,
        },
    }
    plot_epsilon_vs_accuracy(results_by_attack)

    empirical_cov = [0.97, 0.96, 0.94, 0.91, 0.87]
    certified = [0.95, 0.94, 0.92, 0.88, 0.83]
    set_sizes = [1.12, 1.21, 1.38, 1.60, 1.89]
    plot_epsilon_vs_coverage(eps, empirical_cov, certified_bound=certified,
                             set_sizes=set_sizes)

    fracs = [0.0, 0.05, 0.10, 0.15, 0.20]
    q_hats = [0.62, 0.65, 0.71, 0.80, 0.92]
    plot_poisoning_vs_calibration(fracs, q_hats, baseline_q_hat=0.62,
                                  coverage_values=[0.96, 0.95, 0.93, 0.90, 0.85])

    plot_ensemble_uncertainty(
        ensemble_sizes=[1, 2, 3, 5, 7, 10],
        mean_uncertainties=[0.18, 0.13, 0.10, 0.07, 0.06, 0.05],
        std_uncertainties=[0.04, 0.03, 0.02, 0.015, 0.01, 0.008],
    )
    print("\nAll robustness curve plots generated.")


if __name__ == "__main__":
    _synthetic_demo()
