"""Combine sealed source, numeric and visual evidence without viewing pixels."""
import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OUT = HERE / 'pixel_001'
REVIEW = PROJECT / 'paper_program/image_review_batches/batch_004'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(sha_a, sha_b):
    result_path = HERE / 'curation_result.json'
    assert not result_path.exists(), 'No overwrite of completed curation'
    qa_path = OUT / 'independent_technical_qa.json'
    qa = read(qa_path)
    assert qa['status'].startswith('passed') and not qa['failures']
    merger_path = PROJECT / 'paper_program/visual_confirmation_v1/pixel_workflow_003/merge_visual_reviews.py'
    spec = importlib.util.spec_from_file_location('sealed_visual_merge', merger_path)
    merger = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merger)
    merger.main(str(REVIEW), sha_a, sha_b)
    visual = read(REVIEW / 'visual_merge.json')
    visual_by_id = {r['id']: r for r in visual['records']}
    selection = read(OUT / 'selection_manifest.json')
    downloaded = {r['blind_id']: r for r in read(OUT / 'download_manifest.json')['records']}
    source_path = HERE / '八项来源复核.json'
    source = {r['ordinal']: r for r in read(source_path)['rows']}
    duplicates = read(OUT / 'near_duplicate_flags.json')
    recalled = {r['a'] for r in duplicates['flags']}
    recalled.update(r['b'] for r in duplicates['flags'] if r['b_group'] == 'current')
    rows = []
    for selected in selection['rows']:
        ident, ordinal = selected['blind_id'], selected['ordinal']
        record, judgment, origin = downloaded.get(ident, {}), visual_by_id.get(ident, {}), source[ordinal]
        reasons = []
        if origin['source_status'] != 'source-supported':
            reasons.append('source_not_supported')
        if record.get('ordinary_single_image') is not True:
            reasons.append('download_or_format_not_accepted')
        if judgment.get('dual_visual_pass') is not True:
            reasons.append('not_all_five_criteria_true_in_both_reviews')
        if ident in recalled:
            reasons.append('near_duplicate_recall_pending')
        attribution = dict(origin['attribution_draft'])
        attribution['modifications'] = record.get('modification', 'No accepted processed image')
        rows.append(dict(id=ident, ordinal=ordinal, candidate_id=selected['metadata']['id'],
            limited_workflow_usable=not reasons, exclusion_reasons=reasons,
            source_status=origin['source_status'], visual=judgment, attribution=attribution,
            source_cluster=selected['metadata']['cluster_id'],
            files={key:record[key] for key in ('original_path','original_sha256','original_sha1',
                'processed_path','processed_sha256','full_path','full_sha256','crop_path','crop_sha256') if key in record}))
    paths = [Path(__file__), merger_path, source_path, qa_path, HERE/'source_decisions.json',
        HERE/'闭包口径说明.json', OUT/'selection_manifest.json', OUT/'freeze.json',
        OUT/'download_manifest.json', OUT/'near_duplicate_flags.json', OUT/'completion.json',
        REVIEW/'packet.json', REVIEW/'review_a.json', REVIEW/'review_b.json', REVIEW/'visual_merge.json']
    result = dict(status='complete_finite_workflow_curation', created_utc=datetime.now(timezone.utc).isoformat(),
        source_reviewed=8, source_supported=sum(r['source_status']=='source-supported' for r in source.values()),
        fixed_pixel_candidates=4, attempted=len(downloaded), ordinary=sum(r.get('ordinary_single_image',False) for r in downloaded.values()),
        dual_visual_pass=visual['dual_pass'], limited_workflow_usable=sum(r['limited_workflow_usable'] for r in rows),
        rows=rows, reserve_ordinals=[46,51,62,70], reserve_pixels=0, confirmation_v1_pixels=0,
        model_calls=0, new_training_runs=0, historical_v1_usable=11, historical_v1_water_usable=2,
        original_v1_counts_unchanged=True, original_v2_counts_unchanged=True,
        automatically_assigned_to_confirmation=False,
        review_scope='Two fresh-context AI visual reviews; source collector did not view or adjudicate pixels. This is finite workflow material, not completed formation confirmation.',
        limitations=['Source support and hashes do not exclude unknown aliases or all unobserved duplicates.',
            'The narrow metadata frame was chosen after the earlier metadata review; its yield is not an unbiased estimate for general water photos.',
            'No capability or communication outcome has been measured on these new pixels.'],
        evidence_hashes={str(p):sha(p) for p in paths})
    result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','source_supported','attempted','dual_visual_pass','limited_workflow_usable')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sha-a', required=True)
    parser.add_argument('--sha-b', required=True)
    args = parser.parse_args()
    main(args.sha_a, args.sha_b)
