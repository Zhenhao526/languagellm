"""Render static figures for the semantic-transfer summary."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


AXES = ("kind_wood_fiber", "length_short_long", "destination_L_R")
AXIS_NAMES = {
    "kind_wood_fiber": "kind\nwood → fiber",
    "length_short_long": "length\nshort → long",
    "destination_L_R": "destination\nL → R",
}
FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    path = Path(path)
    require(not path.exists(), "Refuse to overwrite " + str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")


def render(summary, out):
    summary = Path(summary).resolve()
    out = Path(out).resolve()
    require(not out.exists(), "Refuse to overwrite figures")
    data = json.loads(summary.read_text(encoding="utf8"))
    require(data["status"] == "completed", "Only completed summary may be plotted")
    require(data["audit"]["status"] == "passed" and data["audit"]["max_abs_error"] == 0.0, "Audit gate failed")
    require(FONT_PATH.is_file(), "Chinese font missing")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_manager.fontManager.addfont(str(FONT_PATH))
    font = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    plt.rcParams.update({"font.family": font, "font.size": 10, "axes.unicode_minus": False, "svg.fonttype": "path", "savefig.facecolor": "white"})
    out.mkdir(parents=True, exist_ok=False)
    files = []
    source_sha = sha(summary)
    script_sha = sha(__file__)

    def save(fig, stem):
        for ext in ("png", "pdf"):
            path = out / f"{stem}.{ext}"
            fig.savefig(path, dpi=220, bbox_inches="tight")
            files.append(path)
        plt.close(fig)

    try:
        # Directional evidence: the silent alias is an exact zero baseline;
        # retain its bar so the route-control interpretation is visible.
        fig, ax = plt.subplots(figsize=(10.5, 6.2))
        x = list(range(len(AXES)))
        width = 0.34
        live = data["by_axis"]["PI_live"]
        silent = data["by_axis"]["PI_silent"]
        live_means = [100 * live[a]["metrics"]["source_pull"]["mean"] for a in AXES]
        live_low = [100 * (live[a]["metrics"]["source_pull"]["mean"] - live[a]["metrics"]["source_pull"]["ci95_t3"][0]) for a in AXES]
        live_high = [100 * (live[a]["metrics"]["source_pull"]["ci95_t3"][1] - live[a]["metrics"]["source_pull"]["mean"]) for a in AXES]
        silent_means = [100 * silent[a]["metrics"]["source_pull"]["mean"] for a in AXES]
        ax.bar([v - width / 2 for v in x], live_means, width, yerr=[live_low, live_high], capsize=4, color="#0072B2", label="PI-live", edgecolor="white", linewidth=0.8)
        ax.bar([v + width / 2 for v in x], silent_means, width, color="#BDBDBD", label="PI-silent（闭路别名）", edgecolor="white", linewidth=0.8)
        ax.axhline(0, color="#333333", linewidth=1.1)
        ax.set_xticks(x, [AXIS_NAMES[a] for a in AXES])
        ax.set_ylabel("source_pull（百分点）")
        ax.set_title("源端消息移植对听者动作集合的定向影响")
        ax.text(0.01, 0.02, "误差线：四个独立种子的描述性 t(3) 95% 区间", transform=ax.transAxes, fontsize=9, color="#555555")
        ax.legend(frameon=False, loc="upper left")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        save(fig, "source_pull_by_axis")

        # Decompose the pull into source/target mass shifts and show behavior
        # change separately from direction of the shift.
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), sharex=True)
        positions = [v - 0.17 for v in x]
        axes[0].bar(positions, [100 * live[a]["metrics"]["delta_source_set_mass"]["mean"] for a in AXES], 0.34, color="#009E73", label="源动作集合质量")
        axes[0].bar([v + 0.17 for v in x], [100 * live[a]["metrics"]["delta_target_set_mass"]["mean"] for a in AXES], 0.34, color="#D55E00", label="目标动作集合质量")
        axes[0].axhline(0, color="#333333", linewidth=1.0)
        axes[0].set_title("动作集合质量变化")
        axes[0].set_ylabel("相对自然路由变化（百分点）")
        axes[0].legend(frameon=False, fontsize=9)
        axes[1].bar(x, [100 * live[a]["metrics"]["delta_action_change_rate"]["mean"] for a in AXES], 0.55, color="#CC79A7")
        axes[1].axhline(0, color="#333333", linewidth=1.0)
        axes[1].set_title("听者贪心动作改变率")
        axes[1].set_ylabel("改变率（百分点）")
        for axis in axes:
            axis.set_xticks(x, [AXIS_NAMES[a] for a in AXES])
            axis.grid(axis="y", alpha=0.2)
            axis.spines[["top", "right"]].set_visible(False)
        fig.suptitle("消息移植的行为改变与方向分解", fontsize=15)
        fig.subplots_adjust(top=0.82, wspace=0.25)
        save(fig, "mass_shift_and_action_change")

        require(sha(summary) == source_sha and sha(__file__) == script_sha, "Figure source changed during rendering")
        receipt = dict(
            status="rendered_pending_visual_review",
            created_at=datetime.now(timezone.utc).isoformat(),
            summary_path=str(summary),
            summary_sha256=source_sha,
            script_sha256=script_sha,
            font_path=str(FONT_PATH),
            font_sha256=sha(FONT_PATH),
            matplotlib_version=matplotlib.__version__,
            files_sha256={path.name: sha(path) for path in files},
            neural_forward_calls=0,
            optimizer_updates=0,
            visual_review="pending",
        )
        write_new(out / "receipt.json", receipt)
        return dict(status=receipt["status"], output=str(out), figures=[path.name for path in files])
    except BaseException as error:
        plt.close("all")
        write_new(out / "failure.json", dict(status="failed", error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.summary, args.out), ensure_ascii=False))
