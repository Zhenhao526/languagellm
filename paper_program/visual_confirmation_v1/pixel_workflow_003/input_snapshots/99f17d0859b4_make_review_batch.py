"""Offline packet split, never changes a download or the frozen collector."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')


def main(mode):
    manifest_bytes = (HERE/'download_manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if mode == 'early':
        first10 = manifest['records'][:10]
        assert len(first10) == 10
        assert all(r['status'] in ('downloaded', 'quarantined_format') for r in first10)
        records = [r for r in first10 if r['status'] == 'downloaded' and r['ordinary_single_image']]
        dest = HERE/'review_early_round1'
        receipt = HERE/'partial_delivery_001'
        prior_ids = set()
        scope = 'Ordinary images available in the first10 fixed-order downloads; delivery during server cooldown. No content selection or replacement. Technical/identity and complete-batch duplicate review pending.'
    else:
        assert manifest['status'] != 'running'
        prior_ids = set(json.loads((HERE/'review_early_round1/packet.json').read_text())['ids'])
        records = [r for r in manifest['records'] if r['status'] == 'downloaded' and r['ordinary_single_image'] and r['blind_id'] not in prior_ids]
        dest = HERE/'review_round2'
        receipt = HERE/'partial_delivery_002'
        scope = 'Remaining successfully downloaded ordinary images not present in early round1. Same criteria; failed or quarantined formats are not silently substituted.'
    dest.mkdir(exist_ok=False)
    receipt.mkdir(exist_ok=False)
    (receipt/'input_manifest_snapshot.json').write_bytes(manifest_bytes)
    ordered = sorted(records, key=lambda r: r['blind_id'])
    assert not prior_ids & {r['blind_id'] for r in ordered}
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 18)
    images, sheets = [], []
    for start in range(0, len(ordered), 4):
        canvas = Image.new('RGB', (1200, 650), 'white')
        draw = ImageDraw.Draw(canvas)
        for j, r in enumerate(ordered[start:start+4]):
            for path_key, digest_key in [('full_path', 'full_sha256'), ('crop_path', 'crop_sha256')]:
                assert sha(r[path_key]) == r[digest_key]
            x, y = j % 2 * 600, j // 2 * 325
            draw.text((x+10, y+8), r['blind_id'], fill='black', font=font)
            full = Image.open(r['full_path']).convert('RGB')
            full.thumbnail((350, 270))
            canvas.paste(full, (x+8+(350-full.width)//2, y+38+(270-full.height)//2))
            canvas.paste(Image.open(r['crop_path']), (x+370, y+55))
            images.append({'id': r['blind_id'], 'full': r['full_path'], 'crop': r['crop_path']})
        path = dest/f'sheet_{start//4+1:02d}.png'
        canvas.save(path)
        sheets.append(str(path))
    packet = {'ids': [r['blind_id'] for r in ordered], 'sheets': sheets, 'images': images,
        'image_sha256': {p: sha(p) for p in sheets},
        'individual_image_sha256': {r[k]: sha(r[k]) for r in ordered for k in ('full_path', 'crop_path')},
        'criteria': ['real photograph', 'food or visible water in transparent container identifiable',
            'crop retains resource', 'no prominent text or watermark', 'no mixed foods or ambiguous liquid'],
        'scope': scope, 'model_scores_available': False}
    write(dest/'packet.json', packet)
    write(receipt/'receipt.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'mode': mode,
        'source_sha256': sha(__file__), 'download_manifest_snapshot_sha256': sha(receipt/'input_manifest_snapshot.json'),
        'packet_sha256': sha(dest/'packet.json'), 'delivered_images': len(ordered), 'sheets': len(sheets),
        'scope': scope, 'new_network_requests': 0, 'image_views': 0, 'model_calls': 0,
        'frozen_collector_sha256': sha(HERE/'collect_pixels.py'), 'frozen_plan_sha256': sha(HERE/'冻结执行方案.md'),
        'no_final_acceptance': True})
    print(json.dumps({'mode': mode, 'images': len(ordered), 'sheets': len(sheets),
        'packet_path': str(dest/'packet.json'), 'packet_sha256': sha(dest/'packet.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['early', 'remaining'])
    main(parser.parse_args().mode)
