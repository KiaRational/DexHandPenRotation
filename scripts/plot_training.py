"""Dark-themed training curves (reward over time etc.) from rsl-rl TensorBoard logs.

    python scripts/plot_training.py --main <run_dir> [--compare <run_dir> ...] --out media/showcase/training_curves.png
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--main", required=True)
p.add_argument("--main_label", default="PPO, 8192 hands")
p.add_argument("--compare", nargs="*", default=[])
p.add_argument("--out", default="media/showcase/training_curves.png")
p.add_argument("--reward_out", default="media/showcase/reward_curve.png")
a = p.parse_args()

PANELS = [  # (title, candidate tag suffixes)
    ("Mean episode reward", ["Train/mean_reward"]),
    ("Spin reward (per step)", ["rew/rotate"]),
    ("Pen turns per episode", ["metric/turns_per_episode"]),
    ("Heading rate [rad/s]", ["metric/heading_rate"]),
    ("Episode length [steps]", ["Train/mean_episode_length"]),
    ("Gravity [m/s²]", ["metric/gravity"]),
]


def load(run):
    ev = sorted(glob.glob(os.path.join(run, "events.out.tfevents.*")))
    data = {}
    for f in ev:
        acc = EventAccumulator(f, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            s = acc.Scalars(tag)
            data.setdefault(tag, []).extend((x.step, x.value) for x in s)
    out = {}
    for tag, v in data.items():
        v = np.array(sorted(dict(v).items()))
        out[tag] = v
    return out


def find(d, suffixes):
    for s in suffixes:
        for tag in d:
            if tag == s or tag.endswith("/" + s) or tag.endswith(s):
                return d[tag]
    return None


def smooth(y, k=50):
    if len(y) < k:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    s = (c[k:] - c[:-k]) / k
    return np.concatenate([np.full(k - 1, np.nan), s])


def style(ax, title):
    ax.set_facecolor("#0d1117")
    ax.set_title(title, color="#e6edf3", fontsize=11)
    ax.tick_params(colors="#8b949e", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    ax.grid(color="#21262d", lw=0.8)
    ax.set_xlabel("PPO iteration", color="#8b949e", fontsize=8)


main = load(a.main)
print("[plot] tags:", sorted(main)[:60])
comps = [(os.path.basename(r).split("_", 2)[-1], load(r)) for r in a.compare]
MAIN_C, CMP_C = "#58e6a8", ["#7aa2ff", "#ff9e64", "#c792ea"]

fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=140)
fig.patch.set_facecolor("#0d1117")
for ax, (title, sfx) in zip(axes.flat, PANELS):
    style(ax, title)
    for (lab, d), c in zip(comps, CMP_C):
        v = find(d, sfx)
        if v is not None:
            ax.plot(v[:, 0], smooth(v[:, 1]), color=c, lw=1.4, alpha=0.75, label=lab)
    v = find(main, sfx)
    if v is not None:
        ax.plot(v[:, 0], v[:, 1], color=MAIN_C, lw=0.6, alpha=0.25)
        ax.plot(v[:, 0], smooth(v[:, 1]), color=MAIN_C, lw=2.2, label=a.main_label)
axes.flat[0].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)
fig.suptitle("Sharpa Wave pen spinning: PPO training in Isaac Lab (8192 parallel hands)", color="#e6edf3", fontsize=14)
fig.tight_layout()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
fig.savefig(a.out, facecolor=fig.get_facecolor())
print("[plot] wrote", a.out)

# single big reward-over-time figure
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
fig.patch.set_facecolor("#0d1117")
style(ax, "Reward over training")
v = find(main, ["Train/mean_reward"])
ax.fill_between(v[:, 0], v[:, 1], np.nanmin(v[:, 1]), color=MAIN_C, alpha=0.08)
ax.plot(v[:, 0], v[:, 1], color=MAIN_C, lw=0.6, alpha=0.3)
ax.plot(v[:, 0], smooth(v[:, 1]), color=MAIN_C, lw=2.4, label="mean episode reward")
ax.set_ylabel("reward", color="#8b949e")
t = find(main, ["metric/turns_per_episode"])
if t is not None:
    ax2 = ax.twinx()
    ax2.plot(t[:, 0], smooth(t[:, 1]), color="#ffd166", lw=1.8, ls="--", label="pen turns / episode")
    ax2.set_ylabel("pen turns per episode", color="#ffd166")
    ax2.tick_params(colors="#ffd166", labelsize=8)
    for sp in ax2.spines.values():
        sp.set_color("#30363d")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", loc="upper left")
fig.tight_layout()
fig.savefig(a.reward_out, facecolor=fig.get_facecolor())
print("[plot] wrote", a.reward_out)
