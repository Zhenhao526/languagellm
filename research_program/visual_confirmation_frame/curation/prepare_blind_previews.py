"""Local technical validation and anonymous, uncropped visual-review copies.

No networks, ML features, labels, source names or prospective split assignments
are placed in the blind review directory. Original byte files remain unchanged.
"""
from pathlib import Path
from datetime import datetime, timezone
from io import BytesIO
import argparse
import hashlib
import json
import warnings

from PIL import Image, ImageOps, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
SALT = 'visual-curation-20260915-v1'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p, algorithm='sha256'):
    return hashlib.new(algorithm, Path(p).read_bytes()).hexdigest()


def write(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def rank(review_id):
    return hashlib.sha256(json.dumps([SALT, 'presentation', review_id], ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def prepare(download, blind, receipt_path):
    download, blind, receipt_path = map(lambda p: Path(p).resolve(), (download, blind, receipt_path))
    assert not blind.exists() and not receipt_path.exists(), 'No overwrite'
    plan = read(download / 'plan.json')
    assert sha(download / 'plan.json') == read(download / 'freeze.json')['plan_sha256']
    result = read(download / 'execution/result.json')
    assert result['status'] == 'download_stage_complete' and result['requests'] == len(plan['jobs'])
    blind.mkdir(parents=True)
    (blind / 'previews').mkdir()
    (blind / 'sheets').mkdir()
    technical, reviewer = [], []
    for job in plan['jobs']:
        receipt = read(download / 'execution/receipts' / f"{job['request_index']:03d}_receipt.json")
        row = {'review_id': job['review_id'], 'pageid': job['pageid'],
               'request_receipt_sha256': sha(download / 'execution/receipts' / f"{job['request_index']:03d}_receipt.json")}
        if receipt['status'] != 'downloaded':
            technical.append(dict(row, status='excluded_before_visual_review', reason=receipt['status']))
            continue
        original = Path(receipt['file'])
        assert sha(original) == receipt['sha256'] and sha(original, 'sha1') == job['original_file_sha1']
        row.update(original_sha256=sha(original), original_sha1=sha(original, 'sha1'))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(original) as source:
                    assert getattr(source, 'n_frames', 1) == 1, 'not_single_frame'
                    assert source.format in ('JPEG', 'PNG', 'TIFF', 'WEBP'), 'not_supported_raster'
                    actual_format = source.format
                    expected = {'JPEG': 'image/jpeg', 'PNG': 'image/png', 'TIFF': 'image/tiff', 'WEBP': 'image/webp'}[actual_format]
                    assert expected == job['mime'], 'mime_disagrees_with_decoded_format'
                    assert source.size == (job['width'], job['height']), 'source_dimensions_differ_from_metadata'
                    row.update(original_mode=source.mode, original_size=list(source.size), format=actual_format,
                               exif_orientation=source.getexif().get(274))
                    oriented = ImageOps.exif_transpose(source)
                    oriented.load()
                    assert min(oriented.size) >= 512, 'oriented_short_side_below_512'
                    row['oriented_size'] = list(oriented.size)
                    # Palette/CMYK files need a displayable PNG mode. This is a
                    # recorded display conversion, never a content alteration.
                    if oriented.mode not in ('RGB', 'RGBA', 'L', 'LA'):
                        oriented = oriented.convert('RGBA' if 'transparency' in oriented.info else 'RGB')
                    oriented.thumbnail((768, 768), resample=Image.Resampling.LANCZOS)
                    preview = blind / 'previews' / (job['review_id'] + '.png')
                    # Re-create pixels so EXIF, title, copyright and other source
                    # metadata do not enter the anonymous reviewer PNG.
                    clean = Image.frombytes(oriented.mode, oriented.size, oriented.tobytes())
                    clean.save(preview, format='PNG')
                    row.update(status='technical_pass_pending_visual_review', preview_sha256=sha(preview),
                               preview_mode=clean.mode, preview_size=list(clean.size),
                               transformations=['EXIF orientation', 'display mode conversion if needed',
                                                'aspect-preserving LANCZOS resize to maximum 768; no crop'])
                    reviewer.append({'review_id': job['review_id'], 'preview_file': str(preview)})
        except Exception as error:
            row.update(status='technical_rejected', reason=repr(error))
        technical.append(row)
    reviewer.sort(key=lambda row: (rank(row['review_id']), row['review_id']))
    write(blind / 'manifest.json', reviewer)
    font = ImageFont.load_default(size=23)
    sheets = []
    for start in range(0, len(reviewer), 6):
        rows = reviewer[start:start + 6]
        sheet = Image.new('RGB', (3 * 788, 2 * 818), 'white')
        draw = ImageDraw.Draw(sheet)
        for j, item in enumerate(rows):
            x, y = (j % 3) * 788, (j // 3) * 818
            draw.text((x + 10, y + 5), item['review_id'], font=font, fill='black')
            with Image.open(item['preview_file']) as im:
                px, py = x + 10 + (768 - im.width) // 2, y + 40 + (768 - im.height) // 2
                if im.mode in ('RGBA', 'LA'):
                    sheet.paste(im, (px, py), im.getchannel('A'))
                else:
                    sheet.paste(im.convert('RGB'), (px, py))
        path = blind / 'sheets' / f'sheet_{start // 6 + 1:02d}.png'
        sheet.save(path)
        sheets.append({'sheet': str(path), 'review_ids': [r['review_id'] for r in rows], 'sha256': sha(path)})
    write(blind / 'sheets.json', sheets)
    report = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(__file__),
              'download_plan_sha256': sha(download / 'plan.json'), 'download_result_sha256': sha(download / 'execution/result.json'),
              'blind_manifest_sha256': sha(blind / 'manifest.json'), 'technical': technical,
              'technical_pass': len(reviewer), 'not_technical_pass': len(technical) - len(reviewer),
              'visual_reviews': 0, 'network_requests': 0, 'model_calls': 0, 'split_assigned': False}
    write(receipt_path, report)
    return {k: report[k] for k in ('technical_pass', 'not_technical_pass', 'visual_reviews', 'network_requests')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', type=Path, default=ROOT / 'water_download_001')
    parser.add_argument('--blind', type=Path, default=ROOT / 'blind_batch_001')
    parser.add_argument('--receipt', type=Path, default=ROOT / 'preview_receipt_001.json')
    args = parser.parse_args()
    print(json.dumps(prepare(args.download, args.blind, args.receipt), indent=2))
