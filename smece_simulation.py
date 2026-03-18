'''
Created on 11 Mar 2026

@author: Dr. Mike

smece_simulation.py
===================
Simulation code for all four SMECE experiments.

Data generating process
-----------------------
1.  x_i ~ Uniform(-3, 3)
2.  ell_i ~ Bernoulli(sigma(k * x_i))          [one binary draw per sample]
3.  Bin x into J equal-width bins over [-3, 3]
4.  y_soft_i = empirical frequency of ell within x-bin of x_i

SMECE bins by p_hat and compares mean(p_hat) vs mean(y_soft) per bin.
ECE   bins by p_hat and compares mean(p_hat) vs mean(ell)    per bin.

Usage
-----
    python smece_simulation.py            # all four experiments, seed=42
    python smece_simulation.py --seed 0   # different seed
    python smece_simulation.py --outdir results/
'''

import argparse
import os
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── constants ─────────────────────────────────────────────────────────────────
MODEL_ORDER = ["A", "B", "C", "D", "E"]
LABELS = {
    "A": "A \u2013 perfect",
    "B": "B \u2013 overconfident",
    "C": "C \u2013 underconfident",
    "D": "D \u2013 biased high",
    "E": "E \u2013 random",
}
COLOURS = {
    "A": "#2166ac", "B": "#d73027",
    "C": "#fc8d59", "D": "#4dac26", "E": "#7b3294",
}
TRUE_RANK  = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
TIED_PAIRS = {frozenset(["B", "C"])}
ALL_PAIRS  = list(combinations(MODEL_ORDER, 2))
K_VALUES   = [0.5, 1, 2, 5, 10, 50]
N_VALUES   = [100, 500, 1000, 5000]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": "#EBEBEB",
    "figure.facecolor": "white",
    "axes.grid": True, "grid.color": "white", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "xtick.bottom": False, "ytick.left": False,
})


# ── data & metrics ────────────────────────────────────────────────────────────
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def generate_data(n, k, J=20, rng=None):
    """
    Draw n samples and compute soft labels via x-binning.

    Returns
    -------
    x        : inputs, shape (n,)
    ell      : binary labels, shape (n,)
    y_soft   : binned empirical frequency soft label, shape (n,)
    p_true   : true latent probability sigma(k*x), shape (n,)
    bin_info : dict  j -> {n, freq, p_true_mean}
    """
    rng    = rng or np.random.default_rng(42)
    x      = rng.uniform(-3, 3, size=n)
    p_true = sigmoid(k * x)
    ell    = rng.binomial(1, p_true)
    edges  = np.linspace(-3, 3, J + 1)
    x_bin  = np.clip(np.searchsorted(edges[1:], x, side="left"), 0, J - 1)
    y_soft = np.zeros(n)
    bin_info = {}
    for j in range(J):
        mask = x_bin == j
        if mask.sum() == 0:
            continue
        freq         = ell[mask].mean()
        y_soft[mask] = freq
        bin_info[j]  = {"n": int(mask.sum()), "freq": freq,
                        "p_true_mean": float(p_true[mask].mean())}
    return x, ell, y_soft, p_true, bin_info


def predict(model_id, x, k, rng=None):
    rng = rng or np.random.default_rng(42)
    if model_id == "A": return sigmoid(k * x)
    if model_id == "B": return sigmoid(3 * k * x)
    if model_id == "C": return sigmoid(0.4 * k * x)
    if model_id == "D": return np.minimum(sigmoid(k * x) + 0.15, 1.0)
    if model_id == "E": return rng.uniform(0, 1, size=len(x))
    raise ValueError(f"Unknown model: {model_id}")


def smece(p_hat, y_soft, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    total, n = 0.0, len(p_hat)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p_hat >= lo) & (p_hat < hi)
        if mask.sum() == 0:
            continue
        total += (mask.sum() / n) * abs(p_hat[mask].mean() - y_soft[mask].mean())
    return total


def ece(p_hat, ell, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    total, n = 0.0, len(p_hat)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p_hat >= lo) & (p_hat < hi)
        if mask.sum() == 0:
            continue
        total += (mask.sum() / n) * abs(p_hat[mask].mean() - ell[mask].mean())
    return total


def correct_ranking(m1, m2, s1, s2):
    if frozenset([m1, m2]) in TIED_PAIRS:
        return True
    return (TRUE_RANK[m1] < TRUE_RANK[m2]) == (s1 < s2)


# ── Experiment 1 ──────────────────────────────────────────────────────────────
def run_experiment1(n=5000, k=2, J=20, rng=None):
    rng = rng or np.random.default_rng(42)
    x, ell, ys, pt, bi = generate_data(n, k, J, rng)
    results = {}
    for m in MODEL_ORDER:
        p = predict(m, x, k, rng)
        results[m] = {"smece": smece(p, ys), "ece": ece(p, ell), "p_hat": p}
    return results, bi


def print_experiment1(results, k, J):
    sv = {m: results[m]["smece"] for m in MODEL_ORDER}
    ev = {m: results[m]["ece"]   for m in MODEL_ORDER}
    sr = {m: r+1 for r, m in enumerate(sorted(MODEL_ORDER, key=lambda m: sv[m]))}
    er = {m: r+1 for r, m in enumerate(sorted(MODEL_ORDER, key=lambda m: ev[m]))}
    print(f"\n{'='*65}\nExperiment 1  k={k}  n=5000  J={J}\n{'='*65}")
    print(f"{'Model':<25} {'SMECE':>8} {'ECE':>8} {'SMECE rank':>11} {'ECE rank':>9}")
    print("-"*65)
    for m in MODEL_ORDER:
        print(f"{LABELS[m]:<25} {sv[m]:>8.4f} {ev[m]:>8.4f} {sr[m]:>11} {er[m]:>9}")


def plot_experiment1(results, k, J, save_path):
    sv   = [results[m]["smece"] for m in MODEL_ORDER]
    ev   = [results[m]["ece"]   for m in MODEL_ORDER]
    cols = [COLOURS[m]          for m in MODEL_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    fig.suptitle(f"Experiment 1: Calibration error  (k={k}, n=5000, J={J} x-bins)",
                 fontsize=11, fontweight="bold", y=1.02)
    for ax, vals, title in zip(axes, [sv, ev],
                                ["SMECE  (soft labels)", "ECE  (raw binary \u2113)"]):
        bars = ax.bar(range(5), vals, color=cols, width=0.58,
                      edgecolor="white", zorder=3)
        ax.set_xticks(range(5))
        ax.set_xticklabels([LABELS[m] for m in MODEL_ORDER],
                            rotation=18, ha="right", fontsize=9)
        ax.set_ylabel("Calibration error", fontsize=10)
        ax.set_title(title, fontsize=10, pad=7)
        ax.set_ylim(0, max(max(sv), max(ev)) * 1.35)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8.2)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Experiment 2 ──────────────────────────────────────────────────────────────
def run_experiment2(n=5000, k_values=None, J=20, rng=None):
    rng = rng or np.random.default_rng(42)
    k_values = k_values or K_VALUES
    smece_k  = {m: [] for m in MODEL_ORDER}
    ece_k    = {m: [] for m in MODEL_ORDER}
    for k in k_values:
        x, ell, ys, _, _ = generate_data(n, k, J, rng)
        for m in MODEL_ORDER:
            p = predict(m, x, k, rng)
            smece_k[m].append(smece(p, ys))
            ece_k[m].append(ece(p, ell))
    return smece_k, ece_k


def print_experiment2(smece_k, ece_k, k_values):
    for label, data in [("SMECE", smece_k), ("ECE", ece_k)]:
        print(f"\n{label}:")
        hdr = f"  {'Model':<23}" + "".join(f"{k:>8}" for k in k_values)
        print(hdr)
        for m in MODEL_ORDER:
            print(f"  {LABELS[m]:<23}" + "".join(f"{v:>8.4f}" for v in data[m]))


def plot_experiment2(smece_k, ece_k, k_values, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Experiment 2: Calibration error vs k  (n=5000, J=20)",
                 fontsize=11, fontweight="bold", y=1.02)
    xp = range(len(k_values))
    for ax, data, title in zip(axes, [smece_k, ece_k], ["SMECE", "ECE"]):
        for m in MODEL_ORDER:
            ax.plot(xp, data[m], color=COLOURS[m], linewidth=2.2,
                    marker="o", markersize=6, label=LABELS[m], zorder=3)
        ax.set_xticks(xp)
        ax.set_xticklabels([str(k) for k in k_values])
        ax.set_xlabel("Label steepness k  (\u2190 softer    harder \u2192)", fontsize=10)
        ax.set_ylabel("Calibration error", fontsize=10)
        ax.set_title(title, fontsize=11, pad=6)
    handles = [Line2D([0],[0], color=COLOURS[m], linewidth=2.2,
                      marker="o", markersize=6, label=LABELS[m])
               for m in MODEL_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.10), frameon=True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Experiment 3 ──────────────────────────────────────────────────────────────
def run_experiment3(n=1000, k_values=None, J=20, n_reps=1000, rng=None):
    rng = rng or np.random.default_rng(42)
    k_values = k_values or K_VALUES
    results  = []
    for k in k_values:
        s_ok = np.zeros(len(ALL_PAIRS))
        e_ok = np.zeros(len(ALL_PAIRS))
        for _ in range(n_reps):
            xr, elr, ysr, _, _ = generate_data(n, k, J, rng)
            ss, es = {}, {}
            for m in MODEL_ORDER:
                p = predict(m, xr, k, rng)
                ss[m] = smece(p, ysr)
                es[m] = ece(p, elr)
            for pi, (m1, m2) in enumerate(ALL_PAIRS):
                s_ok[pi] += correct_ranking(m1, m2, ss[m1], ss[m2])
                e_ok[pi] += correct_ranking(m1, m2, es[m1], es[m2])
        results.append({
            "k": k,
            "smece_overall": s_ok.sum() / (n_reps * len(ALL_PAIRS)),
            "ece_overall":   e_ok.sum() / (n_reps * len(ALL_PAIRS)),
            "smece_pairs":   s_ok / n_reps,
            "ece_pairs":     e_ok / n_reps,
        })
        print(f"  k={k:>4}  SMECE={results[-1]['smece_overall']:.3f}"
              f"  ECE={results[-1]['ece_overall']:.3f}")
    return results


def plot_experiment3(results, k_values, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Experiment 3: Ranking consistency  (1000 reps, n=1000, J=20)",
                 fontsize=11, fontweight="bold", y=1.02)
    xp = range(len(k_values))
    ax = axes[0]
    ax.plot(xp, [r["smece_overall"] for r in results],
            color="#2166ac", linewidth=2.4, marker="o", markersize=8, label="SMECE")
    ax.plot(xp, [r["ece_overall"]   for r in results],
            color="#d73027", linewidth=2.4, marker="s", markersize=8, label="ECE")
    ax.axhline(1.0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xticks(xp); ax.set_xticklabels([str(k) for k in k_values])
    ax.set_xlabel("Label steepness k", fontsize=10)
    ax.set_ylabel("Fraction of pairs correctly ranked", fontsize=10)
    ax.set_title("Overall ranking accuracy (all 10 pairs)", fontsize=10)
    ax.set_ylim(0.55, 1.08)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.legend(fontsize=9)
    r2   = next(r for r in results if r["k"] == 2)
    yp   = np.arange(len(ALL_PAIRS))
    ax2  = axes[1]
    ax2.barh(yp + 0.175, r2["smece_pairs"], 0.35,
             color="#2166ac", alpha=0.85, label="SMECE")
    ax2.barh(yp - 0.175, r2["ece_pairs"],   0.35,
             color="#d73027", alpha=0.85, label="ECE")
    ax2.axvline(1.0, color="#888888", linewidth=0.8, linestyle="--")
    ax2.set_yticks(yp)
    ax2.set_yticklabels([f"{m1} vs {m2}" for m1, m2 in ALL_PAIRS], fontsize=9)
    ax2.set_xlabel("Fraction correctly ranked", fontsize=10)
    ax2.set_title("Per-pair accuracy at k=2", fontsize=10)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax2.set_xlim(0, 1.12)
    ax2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Experiment 4 ──────────────────────────────────────────────────────────────
def run_experiment4(n_values=None, k=2, J=20, n_reps=500, rng=None):
    rng = rng or np.random.default_rng(42)
    n_values = n_values or N_VALUES
    results  = []
    for n in n_values:
        sr = {m: [] for m in MODEL_ORDER}
        er = {m: [] for m in MODEL_ORDER}
        soft_errs = []
        for _ in range(n_reps):
            xr, elr, ysr, ptr, binf = generate_data(n, k, J, rng)
            soft_errs.append(np.mean([
                abs(v["freq"] - v["p_true_mean"]) for v in binf.values()
            ]))
            for m in MODEL_ORDER:
                p = predict(m, xr, k, rng)
                sr[m].append(smece(p, ysr))
                er[m].append(ece(p, elr))
        results.append({
            "n":             n,
            "smece_mean":    {m: np.mean(sr[m]) for m in MODEL_ORDER},
            "smece_std":     {m: np.std(sr[m])  for m in MODEL_ORDER},
            "ece_mean":      {m: np.mean(er[m]) for m in MODEL_ORDER},
            "ece_std":       {m: np.std(er[m])  for m in MODEL_ORDER},
            "soft_err_mean": np.mean(soft_errs),
            "soft_err_std":  np.std(soft_errs),
        })
        print(f"  n={n:>5}  soft-label MAE={results[-1]['soft_err_mean']:.4f}"
              f"  SMECE_A={results[-1]['smece_mean']['A']:.4f}"
              f"\u00b1{results[-1]['smece_std']['A']:.4f}")
    return results


def print_experiment4(results):
    hdr = f"  {'Model':<23}" + "".join(f"{'n='+str(r['n']):>16}" for r in results)
    for label, mk, sk in [("SMECE", "smece_mean", "smece_std"),
                           ("ECE",   "ece_mean",   "ece_std")]:
        print(f"\n{label} mean \u00b1 std:")
        print(hdr)
        for m in MODEL_ORDER:
            row = f"  {LABELS[m]:<23}"
            for r in results:
                row += f"  {r[mk][m]:.3f}\u00b1{r[sk][m]:.3f}"
            print(row)
    print("\nSoft-label estimation error:")
    for r in results:
        print(f"  n={r['n']:>5}: {r['soft_err_mean']:.4f} \u00b1 {r['soft_err_std']:.4f}")


def plot_experiment4(results, save_path):
    n_vals = [r["n"] for r in results]
    xp     = range(len(n_vals))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Experiment 4: Finite-sample convergence  (k=2, J=20, 500 reps)",
        fontsize=11, fontweight="bold", y=1.02)
    panels = [
        (axes[0,0], "smece_mean", "smece_std", "SMECE \u2014 Mean \u00b1 1 SD"),
        (axes[0,1], "ece_mean",   "ece_std",   "ECE \u2014 Mean \u00b1 1 SD"),
        (axes[1,0], "smece_std",  None,         "SMECE \u2014 Std"),
        (axes[1,1], "ece_std",    None,         "ECE \u2014 Std"),
    ]
    for ax, km, ks, title in panels:
        for m in MODEL_ORDER:
            mu = [r[km][m] for r in results]
            ax.plot(xp, mu, color=COLOURS[m], linewidth=2.0,
                    marker="o", markersize=6, label=LABELS[m], zorder=3)
            if ks:
                sg = [r[ks][m] for r in results]
                ax.fill_between(xp,
                                [a - b for a, b in zip(mu, sg)],
                                [a + b for a, b in zip(mu, sg)],
                                color=COLOURS[m], alpha=0.12, zorder=2)
        ax.set_xticks(xp)
        ax.set_xticklabels([f"n={n}" for n in n_vals], fontsize=9)
        ax.set_xlabel("Sample size n", fontsize=10)
        ax.set_ylabel("Calibration error" if ks else "Std deviation", fontsize=10)
        ax.set_title(title, fontsize=10, pad=6)
        ax.legend(fontsize=8, loc="upper right")
    # inset: soft-label estimation error
    ax_in = axes[1,0].inset_axes([0.52, 0.45, 0.45, 0.48])
    ax_in.errorbar(xp,
                   [r["soft_err_mean"] for r in results],
                   yerr=[r["soft_err_std"] for r in results],
                   color="#444444", linewidth=1.8, marker="^",
                   markersize=6, capsize=3)
    ax_in.set_xticks(xp)
    ax_in.set_xticklabels([str(n) for n in n_vals], fontsize=7)
    ax_in.set_title("Soft-label MAE", fontsize=7.5)
    ax_in.tick_params(labelsize=7)
    ax_in.set_facecolor("#F0F0F0")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved: {save_path}")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--outdir", type=str, default="figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"SMECE simulation  seed={args.seed}  output={args.outdir}/\n")

    print("[Experiment 1]")
    r1, bi1 = run_experiment1(n=5000, k=2, J=20, rng=rng)
    print_experiment1(r1, k=2, J=20)
    plot_experiment1(r1, k=2, J=20,
                     save_path=f"{args.outdir}/exp1_bar.pdf")

    print("\n[Experiment 2]")
    s2, e2 = run_experiment2(n=5000, k_values=K_VALUES, J=20, rng=rng)
    print_experiment2(s2, e2, K_VALUES)
    plot_experiment2(s2, e2, K_VALUES,
                     save_path=f"{args.outdir}/exp2_lines.pdf")

    print("\n[Experiment 3]")
    r3 = run_experiment3(n=1000, k_values=K_VALUES, J=20, n_reps=1000, rng=rng)
    plot_experiment3(r3, K_VALUES,
                     save_path=f"{args.outdir}/exp3_ranking.pdf")

    print("\n[Experiment 4]")
    r4 = run_experiment4(n_values=N_VALUES, k=2, J=20, n_reps=500, rng=rng)
    print_experiment4(r4)
    plot_experiment4(r4, save_path=f"{args.outdir}/exp4_convergence.pdf")

    print(f"\nDone.")

