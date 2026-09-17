"""Plot every fixed monitoring trajectory; no model loading or checkpoint selection."""
from pathlib import Path
import argparse
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.run / "execution/results.json"
    data = json.loads(source.read_text())
    assert data["status"] == "completed" and len(data["seeds"]) == 4
    args.out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    colors = ("#0072B2", "#D55E00", "#009E73", "#7A5195")
    names = (("train", "Training states"), ("new_needs", "Unseen need combinations"),
             ("new_layouts", "Unseen layouts"), ("new_needs_and_layouts", "Unseen needs and layouts"))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    handles = []
    for ax, (name, title) in zip(axes.flat, names):
        for seed, color in zip(data["seeds"], colors):
            rows = seed["monitor"]
            x = [r["update"] for r in rows]
            assert x == [0, 100, 500, 1500, 3000, 6000]
            assert all(r["monitor"][name]["worlds"] == 1024 for r in rows)
            y = [r["monitor"][name]["greedy_full_success_rate"] for r in rows]
            line, = ax.plot(x, y, color=color, marker="o", markersize=3.8, linewidth=1.6,
                            label=str(seed["seed"]))
            if name == "train":
                handles.append(line)
        threshold = ax.axhline(.99, color="#666666", linestyle=(0, (4, 3)), linewidth=1,
                              label="99% candidate screen")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.035)
        ax.set_xlim(-80, 6150)
        ax.set_xticks((0, 1500, 3000, 4500, 6000))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", color="#E3E3E3", linewidth=.6)
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel("Full success: independent argmax")
    for ax in axes[-1]:
        ax.set_xlabel("Training updates")
    fig.suptitle("Triadic task: full-information capacity control", x=.08, ha="left", fontsize=16)
    fig.legend(handles=handles + [threshold], loc="upper left", bbox_to_anchor=(.077, .937),
               ncol=5, frameon=False, fontsize=9)
    fig.text(.08, .025, "All four seeds; fixed 1,024-state monitoring sets. Final tables use all 143,424 states per seed.\n"
             "Centralized exact-return training; no communication channel or language-formation claim.",
             fontsize=9, color="#444444", va="bottom")
    fig.subplots_adjust(left=.09, right=.98, top=.84, bottom=.15, hspace=.26, wspace=.14)
    for suffix in ("png", "svg"):
        fig.savefig(args.out / f"learning_trajectories.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = {"source_results_sha256": digest(source), "plot_code_sha256": digest(Path(__file__)),
               "matplotlib": matplotlib.__version__, "all_seeds_included": True,
               "source_measure": "greedy_full_success_rate on fixed monitoring states",
               "files": {p.name: digest(p) for p in args.out.iterdir()}}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": str(args.out), "files": list(receipt["files"])}))


if __name__ == "__main__":
    main()
