"""Scientific schematic of one resource-coordination trial; no training outputs.

Run with the project's .venv/bin/python. Existing CC0/public-domain photos are
embedded without semantic edits; provenance is copied from the dataset manifest.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
V04 = HERE.parents[1]
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['PingFang SC', 'Arial Unicode MS', 'DejaVu Sans'],
    'font.size': 14,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.unicode_minus': False,
})


def main():
    metadata = json.loads((V04 / 'data/manifest.json').read_text())
    entries = {image['id']: image for image in metadata['images']}
    image_ids = ['93ae11be10dd', '5e1694f2142c', '30e6b4759dff', '8566d80e8f59']
    fig = plt.figure(figsize=(10, 9.6), dpi=260, facecolor='white')
    ax = fig.add_axes([.02, .02, .96, .96])
    ax.set(xlim=(0, 10), ylim=(0, 10))
    ax.axis('off')
    dark, gray, message = '#262626', '#747474', '#365D7E'

    def text(x, y, label, size=14, weight='normal', **kw):
        return ax.text(x, y, label, ha='center', va='center', fontsize=size,
                       color=dark, weight=weight, linespacing=1.4, **kw)

    def arrow(start, end, color=gray, lw=1.4, **kwargs):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=15,
                                    linewidth=lw, color=color, shrinkA=3, shrinkB=3, **kwargs))

    def box(x, y, width, height, fill='white', edge='#C9C9C9'):
        ax.add_patch(Rectangle((x, y), width, height, facecolor=fill,
                               edgecolor=edge, linewidth=1.05))

    text(5, 9.73, '一轮私人资源协调任务', size=19, weight='bold')
    text(2.5, 9.17, '主体 A', size=17, weight='bold')
    text(7.5, 9.17, '主体 B', size=17, weight='bold')
    text(2.5, 8.78, '只看自己的两张照片', size=12)
    text(7.5, 8.78, '只看自己的两张照片', size=12)
    image_extents = [(1.02, 2.36), (2.64, 3.98), (6.02, 7.36), (7.64, 8.98)]
    labels = ['食物', '食物', '食物', '水']
    for image_id, (left, right), label in zip(image_ids, image_extents, labels):
        image = Image.open(V04 / entries[image_id]['path']).convert('RGB')
        # Fit the complete photograph; no crop or aspect-ratio distortion.
        image_width, image_height = image.size
        width = right - left
        height = width * image_height / image_width / (9.6 / 10)
        if height > 1.55:
            width *= 1.55 / height
            height = 1.55
        cx = (left + right) / 2
        bottom, top = 7.73 - height / 2, 7.73 + height / 2
        ax.imshow(image, extent=(cx - width / 2, cx + width / 2, bottom, top), aspect='auto', zorder=2)
        ax.add_patch(Rectangle((cx - width / 2, bottom), width, height, fill=False,
                               edgecolor='#D9D9D9', linewidth=.7, zorder=3))
        text(cx, 6.69, label, size=12)
    for center in [2.5, 7.5]:
        arrow((center - .7, 6.6), (center, 6.28))
        arrow((center + .7, 6.6), (center, 6.28))
        box(center - 1.84, 5.66, 3.68, .60, fill='#F3F3F3')
        text(center, 5.96, '冻结 DINOv2 图像表征', size=14)
        arrow((center, 5.66), (center, 5.37))
        box(center - 1.84, 2.47, 3.68, 2.88)
    text(2.5, 5.03, '独立接口 A', size=15, weight='bold')
    text(7.5, 5.03, '独立接口 B', size=15, weight='bold')
    text(2.5, 4.48, '发出 1 个编号 m_A', size=14)
    text(7.5, 4.48, '发出 1 个编号 m_B', size=14)
    text(2.5, 4.12, '五选一  {0, 1, 2, 3, 4}', size=12)
    text(7.5, 4.12, '五选一  {0, 1, 2, 3, 4}', size=12)
    text(5, 4.90, '同时发送', size=11)
    arrow((4.22, 4.53), (5.80, 4.53), color=message, lw=1.9)
    text(5, 4.35, 'm_A', size=11)
    arrow((5.80, 3.93), (4.22, 3.93), color=message, lw=1.9)
    text(5, 3.74, 'm_B', size=11)
    for center, received, choice in [(2.5, 'm_B', '本例采集食物'), (7.5, 'm_A', '本例采集水')]:
        arrow((center, 3.96), (center, 3.55), lw=1.1)
        text(center, 3.30, f'收到 {received} 后\n结合自己的照片选择左或右', size=13)
        text(center, 2.74, choice, size=14, weight='bold')
    arrow((2.5, 2.47), (4.25, 2.02), lw=1.5)
    arrow((7.5, 2.47), (5.75, 2.02), lw=1.5)
    box(3.15, 1.08, 3.70, .92, fill='#F3F3F3')
    text(5, 1.54, '食物 + 水\n双方回报均为 1', size=15, weight='bold')
    text(5, .72, '当轮结算并消耗，不跨轮存储', size=12)
    text(5, .29, '类别名称与文字仅供读图，不进入模型；所示为一个成功选择示例', size=10.5)
    fig.savefig(HERE / 'task_design.png', dpi=260, bbox_inches='tight', pad_inches=.10)
    fig.savefig(HERE / 'task_design.pdf', bbox_inches='tight', pad_inches=.10)
    plt.close(fig)
    credits = []
    for image_id in image_ids:
        entry = entries[image_id]
        credits.append({key: entry.get(key) for key in ['id', 'title', 'author', 'source_url', 'license', 'license_url', 'path']})
    (HERE / 'task_design_credits.json').write_text(json.dumps(credits, ensure_ascii=False, indent=2))
    print('Created task_design.png and task_design.pdf; photo provenance saved')


if __name__ == '__main__':
    main()
