"""Tables and figures for the report, from results/*.npz (run_experiments.py).

Prints the table numbers and writes figures to ../report/figures/.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.colors import ListedColormap

from baseline import rule_policy
from dp import value_iteration
from evaluate import evaluate
from run_experiments import BUDGET, PENALTIES, SEEDS

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})

METHODS = ["monte_carlo", "sarsa", "q_learning", "dqn"]
NAME = {"monte_carlo": "Monte Carlo", "sarsa": "SARSA", "q_learning": "Q-learning", "dqn": "DQN"}
COLOR = {"monte_carlo": "#2a78d6", "sarsa": "#eb6834", "q_learning": "#1baf7a", "dqn": "#eda100"}
SETTINGS = {"decayed": "Decayed", "constant": "Constant 0.10", "reset": "Reset to 0.5"}
SETTING_COLOR = {"decayed": "#2a78d6", "constant": "#eb6834", "reset": "#1baf7a"}
ACTIONS = ListedColormap(["#1baf7a", "#d9d8d4", "#eb6834"])  # Charge, Hold, Discharge
STEPS = {"monte_carlo": 628}  # environment steps per episode; 168 for the others
FIG = "../report/figures"
T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}  # t-value for a 95% interval, by degrees of freedom


def load(method, penalty):
    return [dict(np.load(f"results/{method}_penalty{penalty}_seed{s}.npz")) for s in SEEDS]


def every(method):
    return BUDGET["dqn" if method == "dqn" else "tabular"][2]


def ci(values):
    """Mean and half-width of the 95% confidence interval over seeds."""
    values = np.asarray(values, dtype=float)
    half = T[len(values) - 1] * values.std(ddof=1) / np.sqrt(len(values))
    return values.mean(), half


def reached(curve):
    """First checkpoint after which the 3-checkpoint moving average of regret stays at or below 2%, or None."""
    curve = np.asarray(curve)
    average = [curve[max(0, k - 2):k + 1].mean() for k in range(len(curve))]
    above = np.nonzero(np.asarray(average) > 2)[0]
    if len(above) == 0:
        return 0
    return None if above[-1] == len(curve) - 1 else above[-1] + 1


def speed(method, curves):
    """Seeds reached, and episodes and steps to reach (over the seeds that reached)."""
    hits = [reached(c) for c in curves]
    episodes = [(k + 1) * every(method) for k in hits if k is not None]
    return len(episodes), episodes, [e * STEPS.get(method, 168) for e in episodes]


def fmt(mean_half, digits=2):
    m, h = mean_half
    return f"{m:,.{digits}f} ± {h:,.{digits}f}"


def effective(policy):
    """Charge at a full battery and Discharge at an empty one act as Hold, so show them as Hold."""
    policy = policy.copy()
    policy[10][policy[10] == 0] = 1
    policy[0][policy[0] == 2] = 1
    return policy


def heatmap(ax, policy, title, differs=None):
    policy = effective(policy)
    ax.imshow(policy[:, :, 0], origin="lower", cmap=ACTIONS, vmin=0, vmax=2, aspect="auto")
    if differs is not None:
        b, h = np.nonzero(differs)
        ax.scatter(h, b, marker="x", s=12, color="#0b0b0b", linewidths=0.8)
    ax.set_title(title)
    ax.set_xlabel("Hour")
    ax.set_xticks(range(0, 24, 4))
    ax.set_yticks(range(0, 11, 2))


def legend_actions(fig):
    handles = [plt.Rectangle((0, 0), 1, 1, color=ACTIONS(i)) for i in range(3)]
    fig.legend(handles, ["Charge", "Hold", "Discharge"], loc="lower center", ncol=4, frameon=False)


def main():
    os.makedirs(FIG, exist_ok=True)
    out = {}
    eval_seeds = [1000 + s for s in SEEDS]
    dp = {(p, k): value_iteration(p, k)[1] for p in ("normal", "summer") for k in PENALTIES}
    rule = rule_policy()

    # Fig. 2: DP policies, grid on
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.4), sharey=True)
    normal, summer = dp[("normal", 50)], dp[("summer", 50)]
    heatmap(axes[0], normal, "(a) Normal pattern")
    heatmap(axes[1], summer, "(b) Summer pattern")
    heatmap(axes[2], summer, "(c) Differences marked", differs=effective(normal)[:, :, 0] != effective(summer)[:, :, 0])
    axes[0].set_ylabel("Battery level")
    legend_actions(fig)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(f"{FIG}/fig2_dp_policies.png")
    out["dp_policy_differences"] = int((effective(normal)[:, :, 0] != effective(summer)[:, :, 0]).sum())

    # Experiment 1 at every penalty: the "before" curve continued by the "normal" branch
    exp1, table11, table12, final = {}, {}, {}, {}
    for k in PENALTIES:
        rule_regret = []
        for s in eval_seeds:
            best = evaluate(dp[("normal", k)], "normal", k, seed=s)[0]
            rule_regret.append((best - evaluate(rule, "normal", k, seed=s)[0]) / -best * 100)
        final[(k, "rule")] = ci(rule_regret)
        for m in METHODS:
            runs = load(m, k)
            curves = [np.concatenate([r["before"], r["normal"]]) for r in runs]
            final[(k, m)] = ci([c[-1] for c in curves])
            if k == 50:
                exp1[m] = (curves, runs)
    for m in METHODS:
        curves, runs = exp1[m]
        scores = np.array([evaluate(r["normal_policy"], "normal", 50, seed=s) for r, s in zip(runs, eval_seeds)])
        table11[m] = [fmt(final[(50, m)])] + [fmt(ci(scores[:, j])) for j in range(3)]
        n, episodes, steps = speed(m, curves)
        table12[m] = [f"{n} / {len(SEEDS)}",
                      fmt(ci(episodes), 0) if n > 1 else (f"{episodes[0]:,}" if n else "–"),
                      fmt(ci(steps), 0) if n > 1 else (f"{steps[0]:,}" if n else "–")]
    for name, policy in (("dp", dp[("normal", 50)]), ("rule", rule)):
        scores = np.array([evaluate(policy, "normal", 50, seed=s) for s in eval_seeds])
        table11[name] = [fmt(final[(50, "rule")]) if name == "rule" else "0 (reference)"] + \
                        [fmt(ci(scores[:, j])) for j in range(3)]
    out["table11"], out["table12"] = table11, table12

    # Fig. 3: Experiment 1 learning curves
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
    for m in METHODS:
        curves = np.array(exp1[m][0])
        episodes = (np.arange(curves.shape[1]) + 1) * every(m)
        mean = curves.mean(axis=0)
        half = T[len(curves) - 1] * curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
        for ax, x in ((axes[0], episodes), (axes[1], episodes * STEPS.get(m, 168))):
            ax.plot(x, mean, color=COLOR[m], lw=1.5, label=NAME[m])
            ax.fill_between(x, mean - half, mean + half, color=COLOR[m], alpha=0.15, lw=0)
    for ax, label in ((axes[0], "(a) Training episodes"), (axes[1], "(b) Environment steps")):
        ax.set_xscale("log")
        ax.axhline(2, color="#52514e", ls="--", lw=0.8)
        ax.axhline(final[(50, "rule")][0], color="#52514e", ls=":", lw=0.8)
        ax.set_xlabel(label)
        ax.grid(alpha=0.2)
    axes[0].text(axes[0].get_xlim()[0] * 1.1, 2.3, "2% level", fontsize=7, color="#52514e")
    axes[0].text(axes[0].get_xlim()[0] * 1.1, final[(50, "rule")][0] - 0.9, "Rule-based", fontsize=7,
                 color="#52514e")
    axes[0].set_ylabel("Regret (%)")
    axes[0].set_ylim(-3, 15)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(f"{FIG}/fig3_experiment1.png")

    # Fig. 4: final greedy policy of the median seed for each method
    fig, axes = plt.subplots(1, 4, figsize=(7, 2.3), sharey=True)
    out["fig4_differences"] = {}
    for ax, m in zip(axes, METHODS):
        curves, runs = exp1[m]
        seed = int(np.argsort([c[-1] for c in curves])[len(curves) // 2])
        policy = runs[seed]["normal_policy"]
        differs = effective(policy)[:, :, 0] != effective(normal)[:, :, 0]
        heatmap(ax, policy, NAME[m], differs=differs)
        out["fig4_differences"][m] = {"seed": seed, "all_seeds": [int((effective(r["normal_policy"])[:, :, 0]
                                      != effective(normal)[:, :, 0]).sum()) for r in runs], "states": int(differs.sum()),
                                      "final_regret": float(curves[seed][-1])}
    axes[0].set_ylabel("Battery level")
    legend_actions(fig)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(f"{FIG}/fig4_learned_policies.png")

    # Experiment 2 (penalty 50): regret against normal DP before the change, summer DP after
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.6))
    table13 = {}
    for ax, m in zip(axes.flat, METHODS):
        runs = load(m, 50)
        n_before = len(runs[0]["before"])
        for setting in SETTINGS:
            before = "constant_before" if setting == "constant" else "before"
            after = np.array([r[setting] for r in runs])
            curves = np.concatenate([np.array([r[before] for r in runs]), after], axis=1)
            episodes = (np.arange(curves.shape[1]) + 1) * every(m)
            mean = curves.mean(axis=0)
            half = T[len(curves) - 1] * curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
            ax.plot(episodes, mean, color=SETTING_COLOR[setting], lw=1.3, label=SETTINGS[setting])
            ax.fill_between(episodes, mean - half, mean + half, color=SETTING_COLOR[setting], alpha=0.15, lw=0)
            n, ep, st = speed(m, after)
            table13[(m, setting)] = [fmt(ci([r[before][-1] for r in runs])), fmt(ci(after[:, 0])),
                                     f"{n} / {len(SEEDS)}",
                                     fmt(ci(ep), 0) if n > 1 else (f"{ep[0]:,}" if n else "–"),
                                     fmt(ci(st), 0) if n > 1 else (f"{st[0]:,}" if n else "–")]
        ax.axvline(n_before * every(m), color="#52514e", lw=0.8)
        ax.axhline(2, color="#52514e", ls="--", lw=0.8)
        ax.set_title(NAME[m] + (" (own scale)" if m == "dqn" else ""))
        ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_ylim(-3, 80 if m == "dqn" else 15)
        ax.grid(alpha=0.2)
    for ax in axes[1]:
        ax.set_xlabel("Training episodes")
    for ax in axes[:, 0]:
        ax.set_ylabel("Regret (%)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(f"{FIG}/fig5_experiment2.png")
    out["table13"] = {f"{m}/{s}": v for (m, s), v in table13.items()}

    # Experiment 3: reference policies (100,000 days, as Table X) and Fig. 6
    out["table14"] = {}
    for k in (20, 100):
        for name, policy in (("DP optimal", dp[("normal", k)]), ("Rule-based", rule)):
            out["table14"][f"{k}/{name}"] = [f"{v:.2f}" for v in evaluate(policy, "normal", k, days=100_000)]
    fig, ax = plt.subplots(figsize=(5, 2.6))
    width = 0.2
    for j, m in enumerate(METHODS):
        means = [final[(k, m)][0] for k in sorted(PENALTIES)]
        halves = [final[(k, m)][1] for k in sorted(PENALTIES)]
        ax.bar(np.arange(3) + (j - 1.5) * width, means, width * 0.9, yerr=halves, color=COLOR[m],
               label=NAME[m], capsize=2, error_kw={"lw": 0.8})
    ax.scatter(np.arange(3), [final[(k, "rule")][0] for k in sorted(PENALTIES)], marker="D", s=18,
               color="#0b0b0b", zorder=3, label="Rule-based")
    top = 12
    for j, m in enumerate(METHODS):
        for i, k in enumerate(sorted(PENALTIES)):
            if final[(k, m)][0] > top:
                ax.text(i + (j - 1.5) * width, top * 0.97, f"{final[(k, m)][0]:.0f}%↑", ha="center", va="top",
                        fontsize=7, color="#0b0b0b")
    ax.set_ylim(-2, top)
    ax.axhline(0, color="#52514e", lw=0.6)
    ax.set_xticks(range(3), [f"κ = {k}" for k in sorted(PENALTIES)])
    ax.set_ylabel("Final regret (%)")
    ax.legend(frameon=False, fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig6_experiment3.png")
    out["final_regret"] = {f"{k}/{m}": fmt(v) for (k, m), v in final.items()}

    print(json.dumps(out, indent=1, ensure_ascii=False))
    with open("results/summary.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
