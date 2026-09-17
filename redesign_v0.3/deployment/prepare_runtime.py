"""Create a minimally adapted copy of the official VPT inference code.

No architecture or tensor math is replaced. The untouched Git checkout remains
the provenance source. MineRL is lazy-imported only by its unused item-name helper.
"""
from pathlib import Path
import difflib
import hashlib
import json
import io
import pickle
import shutil
import subprocess

BASE = Path(__file__).resolve().parent


def main():
    source = BASE / 'Video-Pre-Training'
    target = BASE / 'runtime'
    if not (source / 'lib/policy.py').exists():
        raise RuntimeError('Clone the official repository first')
    target.mkdir(exist_ok=True)
    shutil.copytree(source / 'lib', target / 'lib', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(source / 'LICENSE', target / 'LICENSE')
    replacements = {
        'lib/actions.py': [
            ('import minerl.herobraine.hero.mc as mc\n', ''),
            ('        return mc.MINERL_ITEM_MAP[item_id]',
             '        import minerl.herobraine.hero.mc as mc\n        return mc.MINERL_ITEM_MAP[item_id]'),
        ],
        'lib/torch_util.py': [('    return th.has_cuda', '    return th.cuda.is_available()')],
    }
    patches = []
    for relative, edits in replacements.items():
        path = target / relative
        before = path.read_text()
        after = before
        for old, new in edits:
            if after.count(old) != 1:
                raise RuntimeError(f'Unexpected upstream text in {relative}: {old}')
            after = after.replace(old, new, 1)
        path.write_text(after)
        patches.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                       fromfile=f'upstream/{relative}', tofile=f'runtime/{relative}'))
    (BASE / 'compatibility.patch').write_text(''.join(patches))
    files = {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(target.rglob('*.py'))}
    provenance = {
        'upstream': 'https://github.com/openai/Video-Pre-Training',
        'commit': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
        'purpose': 'Pretrained policy inference only; no Minecraft environment is installed by this script.',
        'adaptations': ['Lazy import for unused MineRL item-name helper',
                        'Replace removed torch.has_cuda with torch.cuda.is_available'],
        'runtime_sha256': files,
    }
    (BASE / 'source_provenance.json').write_text(json.dumps(provenance, indent=2))
    model_file = BASE / 'weights/2x.model'
    if model_file.exists():
        from gym3 import types
        allowed = {('gym3.types', name): getattr(types, name)
                   for name in ('DictType', 'TensorType', 'Discrete')}
        class RestrictedConfigUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if (module, name) not in allowed:
                    raise pickle.UnpicklingError(f'Unexpected config global: {module}.{name}')
                return allowed[(module, name)]
        config = RestrictedConfigUnpickler(io.BytesIO(model_file.read_bytes())).load()
        (BASE / 'weights/2x.config.json').write_text(json.dumps(config, default=str, indent=2))
    print(json.dumps({'runtime': str(target), 'upstream_commit': provenance['commit']}))


if __name__ == '__main__':
    main()
