"""Run the unmodified radar CLI on the transparent five-paper arXiv subset.

For Luna and Kouwenhoven the cache contains the downloaded formal ACL version,
not an assertion of byte identity with the arXiv version. Provenance is explicit.
Author-enrichment timeouts are retained and do not substitute for content review.
"""
from pathlib import Path
import subprocess, json, shutil, hashlib, sys

B = Path(__file__).resolve().parent
RADAR = '/Users/xia/.codex/plugins/cache/zhenhao-arxiv-tools/arxiv-watcher/0.2.0/skills/arxiv-paper-radar/scripts/radar.py'
FILES = {
    '2004.03868': ('2020_RodriguezLuna_Object_Constancy_FindingsEMNLP.pdf', 'Formal Findings EMNLP PDF; arXiv entry is used only for metadata'),
    '2002.01335': ('2021_Slowik_Structural_Inductive_Biases_arXiv2002.01335v4.pdf', 'arXiv v4 PDF'),
    '2305.10920': ('2023_Ri_Emergent_Communication_with_Attention_arXiv2305.10920v1.pdf', 'arXiv v1 PDF'),
    '2407.17960': ('2024_Kouwenhoven_Representational_Alignment_CMCL.pdf', 'Formal CMCL PDF; arXiv entry is used only for metadata'),
    '2605.27532': ('2026_Abouelyazid_SCALE_COMM_arXiv2605.27532v1.pdf', 'arXiv v1 PDF'),
}

def main():
    receipts = []
    (B/'packets').mkdir(exist_ok=True)
    for aid, (filename, version_note) in FILES.items():
        source = B/'primary'/filename
        target = B/'data/pdfs'/f'{aid}.pdf'
        shutil.copyfile(source, target)
        receipt = {'arxiv_id': aid, 'packet_pdf_source': str(source.relative_to(B)),
                   'version_note': version_note, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
        cmd = [sys.executable, RADAR, 'packet', aid, '--config', str(B/'config.toml'),
               '--max-chars', '200000', '--output', str(B/'packets'/f'{aid}.md')]
        with (B/'logs'/f'packet_{aid}.txt').open('w') as log:
            try:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=100)
                receipt['exit_code'] = result.returncode
            except subprocess.TimeoutExpired:
                receipt['status'] = 'timeout_100s_preserved_primary_fulltext'
        receipt['packet_exists'] = (B/'packets'/f'{aid}.md').exists()
        receipts.append(receipt)
        (B/'packet_provenance.json').write_text(json.dumps(receipts, ensure_ascii=False, indent=2)+'\n')
        print(aid, receipt.get('exit_code', receipt.get('status')), flush=True)

if __name__ == '__main__':
    main()
