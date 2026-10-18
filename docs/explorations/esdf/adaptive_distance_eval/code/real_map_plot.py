"""Standalone scientific figures from saved real-map evidence, in Docker."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np


def footprint(ax, boxes, bounds=None):
    patches = []
    for lo, hi in boxes:
        if bounds is not None and ((hi[:2] < bounds[0]).any() or (lo[:2] > bounds[1]).any()):
            continue
        patches.append(Rectangle(lo[:2], *(hi[:2] - lo[:2])))
    ax.add_collection(PatchCollection(patches, facecolor="#657581", alpha=.35, edgecolor="none"))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not Path("/.dockerenv").exists():
        raise RuntimeError("run inside Docker")
    output = args.output
    result = json.loads((output / "results.json").read_text())
    raw = np.load(output / "optimization/batch-optimization.npz")
    boxes = np.asarray(result["manifest"]["geometry"]["boxes"])
    before, after = raw["before_position"], raw["after_position"]
    minimum_before = raw["before_distance"].min(1) - .3
    minimum_after = raw["after_distance"].min(1) - .3
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    ax = axes[0, 0]
    footprint(ax, boxes)
    for index in range(len(before)):
        ax.plot(before[index, :, 0], before[index, :, 1], color="#d28332", lw=.8, alpha=.6)
        ax.plot(after[index, :, 0], after[index, :, 1], color="#156a9b", lw=1., alpha=.8)
    ax.plot([], [], color="#d28332", label="Before")
    ax.plot([], [], color="#156a9b", label="After")
    ax.set(xlim=(-30, 30), ylim=(-30, 30), title=f"500-pillar map / {len(before)} trajectories (XY projection)")
    ax.legend(loc="upper left")

    ax = axes[0, 1]
    for name, color in (("full", "#156a9b"), ("roi-coarse", "#d28332"), ("roi-adaptive", "#318770")):
        data = np.load(output / f"{name}-accuracy.npz")
        errors = np.sort(np.maximum(np.abs(data["predicted"]-data["expected"])*1000, 1e-5))
        threshold = 1000*result["manifest"]["configurations"][name]["termination_error_m"]
        config = result["manifest"]["configurations"][name]
        minimum = np.ptp(np.asarray(config["bounds"]), axis=0)[0]/2**config["max_depth"]
        label = "ROI fixed 0.2 m" if name == "roi-coarse" else f"{name} (min {minimum:g} m; {threshold:g} mm heuristic)"
        ax.plot(errors, np.arange(1, len(errors)+1)/len(errors), color=color, label=label)
    ax.axvline(1., color="#888888", ls=":", lw=1)
    ax.set(xscale="log", xlabel="Absolute distance error (mm)", ylabel="Empirical fraction",
           ylim=(0., 1.01), title="Independent probes (different full / ROI domains)\nfloat64 evaluation of stored float32 fields")
    ax.legend()
    ax.grid(alpha=.2)

    ax = axes[1, 0]
    colors = ("#b14e44", "#d28332", "#318770")
    names = raw["group_names"]
    for group in range(3):
        mask = raw["group"] == group
        ax.scatter(minimum_before[mask], minimum_after[mask], s=30, color=colors[group], label=str(names[group]), alpha=.8)
    limits = [min(minimum_before.min(), minimum_after.min())-.05,
              max(minimum_before.max(), minimum_after.max())+.05]
    ax.plot(limits, limits, color="#888888", ls="--", lw=1)
    ax.axhline(0., color="#888888", lw=.7)
    ax.axvline(0., color="#888888", lw=.7)
    ax.set(xlim=limits, ylim=limits, xlabel="Before: observed distance - 0.3 m",
           ylabel="After: observed distance - 0.3 m", title="513-time independent probes; above diagonal improves")
    ax.legend()
    ax.grid(alpha=.2)

    ax = axes[1, 1]
    for key, label, color in (("main_cost", "Total cost", "#156a9b"), ("main_obstacle", "Obstacle cost", "#d28332")):
        values = raw[key]
        values = values / np.maximum(values[:1], 1e-12)
        ax.plot(np.median(values, axis=1), color=color, label=label)
        ax.fill_between(np.arange(len(values)), np.quantile(values, .1, axis=1),
                        np.quantile(values, .9, axis=1), color=color, alpha=.15)
    ax.set(xlabel="Accepted/rejected update round", ylabel="Ratio to initial value",
           title="Median and P10-P90; total descent enforced by line search")
    ax.legend()
    ax.grid(alpha=.2)
    fig.suptitle("Shared-node adaptive SdfLib: actual-map experiment", fontsize=15)
    fig.savefig(output / "real-map-results.png", dpi=160)
    plt.close(fig)

    indices, labels = [], []
    for index, label in ((int(np.argmin(minimum_before)), "Initial worst"),
                         (int(np.argmax(minimum_after-minimum_before)), "Largest gain"),
                         (int(np.argmin(minimum_after)), "Final worst")):
        if index not in indices:
            indices.append(index)
            labels.append(label)
    while len(indices) < 3:
        indices.append(next(i for i in range(len(before)) if i not in indices))
        labels.append("Additional task")
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True)
    for column, index in enumerate(indices):
        lo = np.minimum(before[index].min(0), after[index].min(0))[:2]-.8
        hi = np.maximum(before[index].max(0), after[index].max(0))[:2]+.8
        ax = axes[0, column]
        footprint(ax, boxes, np.array([lo, hi]))
        ax.plot(before[index, :, 0], before[index, :, 1], color="#d28332", ls="--", label="Before")
        ax.plot(after[index, :, 0], after[index, :, 1], color="#156a9b", label="After")
        ax.scatter(*before[index, 0, :2], color="black", s=20)
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]),
               title=f"{labels[column]} / task {index}: surplus {minimum_before[index]:.3f} -> {minimum_after[index]:.3f} m")
        ax.legend()
        ax = axes[1, column]
        ax.plot(raw["before_fractions"], before[index, :, 2], color="#d28332", ls="--", label="Before")
        ax.plot(raw["after_fractions"], after[index, :, 2], color="#156a9b", label="After")
        ax.set(xlabel="Normalized time", ylabel="World Z (m)")
        ax.grid(alpha=.2)
        ax = axes[2, column]
        ax.plot(raw["before_fractions"], raw["before_distance"][index], color="#d28332", ls="--", label="Before")
        ax.plot(raw["after_fractions"], raw["after_distance"][index], color="#156a9b", label="After")
        ax.axhline(.3, color="#318770", lw=.8, ls=":", label="Required 0.3 m")
        ax.axhline(.2, color="#888888", lw=.7, ls=":", label="Body radius 0.2 m")
        ax.axhline(0., color="#888888", lw=.7)
        ax.set(xlabel="Normalized time", ylabel="Observed mesh distance (m)")
        ax.grid(alpha=.2)
        if column == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Actual-map examples: XY footprints / Z / independent distance; surplus = distance - 0.3 m", fontsize=13)
    fig.savefig(output / "trajectory-examples.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"plots": [str(output / "real-map-results.png"), str(output / "trajectory-examples.png")]}))


if __name__ == "__main__":
    main()
