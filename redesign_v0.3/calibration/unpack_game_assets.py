"""Copy publisher-hashed game assets into the official MineRL resource layout."""
from pathlib import Path
import hashlib
import json
import shutil

BASE=Path(__file__).resolve().parent


def main():
    cache=BASE/'runtime/gradle-cache/caches/forge_gradle/assets'
    index=cache/'indexes/1.16.json'
    objects=json.loads(index.read_text())['objects']
    destination=BASE/'vendor/minerl/minerl/MCP-Reborn/src/main/resources'
    count=total=0
    for name,meta in objects.items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Unexpected asset path')
        source=cache/'objects'/meta['hash'][:2]/meta['hash']
        data=source.read_bytes()
        if len(data)!=meta['size'] or hashlib.sha1(data).hexdigest()!=meta['hash']:
            raise RuntimeError(f'Publisher asset hash mismatch: {name}')
        target=destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        count+=1
        total+=len(data)
    receipt={'count':count,'bytes':total,'all_publisher_sha1_verified':True,
             'index_sha256':hashlib.sha256(index.read_bytes()).hexdigest()}
    (BASE/'logs/assets_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
