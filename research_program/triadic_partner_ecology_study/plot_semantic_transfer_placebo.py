"""Render aligned versus misaligned-message placebo figures."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


AXES = ("kind_wood_fiber", "length_short_long", "destination_L_R")
NAMES = {"kind_wood_fiber": "kind\nwood → fiber", "length_short_long": "length\nshort → long", "destination_L_R": "destination\nL → R"}
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
    require(not Path(path).exists(), "Refuse to overwrite " + str(path))
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")


def render(summary, out):
    summary = Path(summary).resolve(); out = Path(out).resolve(); require(not out.exists(), "Refuse to overwrite figures")
    data = json.loads(summary.read_text(encoding="utf8")); require(data["status"] == "completed" and data["audit"]["status"] == "passed", "Summary/audit gate failed")
    require(FONT_PATH.is_file(), "Chinese font missing")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(FONT_PATH)); font = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    plt.rcParams.update({"font.family": font, "font.size": 10, "axes.unicode_minus": False, "svg.fonttype": "path", "savefig.facecolor": "white"})
    out.mkdir(parents=True); files = []; source_sha = sha(summary); script_sha = sha(__file__)

    def save(fig, stem):
        for ext in ("png", "pdf"):
            path = out / f"{stem}.{ext}"; fig.savefig(path, dpi=220, bbox_inches="tight"); files.append(path)
        plt.close(fig)

    try:
        fig, ax = plt.subplots(figsize=(10.8, 6.2)); x = list(range(len(AXES))); width = 0.34
        aligned = data["by_axis"]["PI_live"]
        am = [100 * aligned[a]["metrics"]["aligned"]["source_pull"]["mean"] for a in AXES]
        pm = [100 * aligned[a]["metrics"]["placebo"]["source_pull"]["mean"] for a in AXES]
        ae = [[100 * (aligned[a]["metrics"]["aligned"]["source_pull"]["mean"] - aligned[a]["metrics"]["aligned"]["source_pull"]["ci95_t3"][0]) for a in AXES], [100 * (aligned[a]["metrics"]["aligned"]["source_pull"]["ci95_t3"][1] - aligned[a]["metrics"]["aligned"]["source_pull"]["mean"]) for a in AXES]]
        pe = [[100 * (aligned[a]["metrics"]["placebo"]["source_pull"]["mean"] - aligned[a]["metrics"]["placebo"]["source_pull"]["ci95_t3"][0]) for a in AXES], [100 * (aligned[a]["metrics"]["placebo"]["source_pull"]["ci95_t3"][1] - aligned[a]["metrics"]["placebo"]["source_pull"]["mean"]) for a in AXES]]
        ax.bar([i - width / 2 for i in x], am, width, yerr=ae, capsize=4, color="#0072B2", label="aligned 源消息")
        ax.bar([i + width / 2 for i in x], pm, width, yerr=pe, capsize=4, color="#D55E00", label="placebo 错配消息")
        ax.axhline(0, color="#333333", linewidth=1.1); ax.set_xticks(x, [NAMES[a] for a in AXES]); ax.set_ylabel("source_pull（百分点）"); ax.set_title("正确源消息与同层错配消息的定向效应")
        ax.text(0.01, 0.02, "误差线：四个独立种子的描述性 t(3) 95% 区间", transform=ax.transAxes, fontsize=9, color="#555555"); ax.legend(frameon=False, loc="upper left"); ax.grid(axis="y", alpha=.2); ax.spines[["top", "right"]].set_visible(False); save(fig, "aligned_vs_placebo_source_pull")

        fig, ax = plt.subplots(figsize=(10.8, 6.0)); am = [100 * aligned[a]["metrics"]["aligned_minus_placebo"]["source_pull"]["mean"] for a in AXES]; ae = [[100 * (aligned[a]["metrics"]["aligned_minus_placebo"]["source_pull"]["mean"] - aligned[a]["metrics"]["aligned_minus_placebo"]["source_pull"]["ci95_t3"][0]) for a in AXES], [100 * (aligned[a]["metrics"]["aligned_minus_placebo"]["source_pull"]["ci95_t3"][1] - aligned[a]["metrics"]["aligned_minus_placebo"]["source_pull"]["mean"]) for a in AXES]]
        ax.bar(x, am, .55, yerr=ae, capsize=4, color="#009E73"); ax.axhline(0, color="#333333", linewidth=1.1); ax.set_xticks(x, [NAMES[a] for a in AXES]); ax.set_ylabel("aligned − placebo（百分点）"); ax.set_title("源消息需求对齐相对于错配 placebo 的增量")
        ax.text(0.01, 0.02, "正值才支持需求对齐；区间为四种子的描述性 t(3) 95% 区间", transform=ax.transAxes, fontsize=9, color="#555555"); ax.grid(axis="y", alpha=.2); ax.spines[["top", "right"]].set_visible(False); save(fig, "aligned_minus_placebo_by_axis")
        require(sha(summary) == source_sha and sha(__file__) == script_sha, "Figure source changed during rendering")
        receipt = dict(status="rendered_pending_visual_review", created_at=datetime.now(timezone.utc).isoformat(), summary_path=str(summary), summary_sha256=source_sha, script_sha256=script_sha, font_path=str(FONT_PATH), font_sha256=sha(FONT_PATH), matplotlib_version=matplotlib.__version__, files_sha256={p.name: sha(p) for p in files}, neural_forward_calls=0, optimizer_updates=0, visual_review="pending")
        write_new(out / "receipt.json", receipt); return dict(status=receipt["status"], output=str(out), figures=[p.name for p in files])
    except BaseException as error:
        plt.close("all"); write_new(out / "failure.json", dict(status="failed", error_type=type(error).__name__, error=str(error))); raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--summary", required=True); parser.add_argument("--out", required=True); args = parser.parse_args(); print(json.dumps(render(args.summary, args.out), ensure_ascii=False))
