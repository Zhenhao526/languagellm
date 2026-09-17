#!/usr/bin/env python3
"""Prepare a small licensed photographic stimulus set, never model inputs with labels.

Metadata comes from Wikimedia's official imageinfo API. English Wikipedia exposes
Commons-hosted media metadata even when the Commons API is temporarily unavailable.
Run download, inspect contact sheets, then finalize with a reviewed selection file.
The model may receive only decoded RGB pixels; all strings below are audit metadata.
"""
from __future__ import annotations
import argparse, concurrent.futures, datetime, hashlib, html, json, re, time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
UA = 'ResourceResearch/0.4 (local non-commercial emergent communication experiment)'
FOOD = '''Apfel 01.jpg
Apfel 6.jpg
Apole.jpg
Apple (2020-12-11).jpg
Apple (511354151).jpg
Apple (Malus domestica) (19719834878).jpg
Apple basket (25963669243).jpg
Apple cluster on a tree branch in Vermont, US.jpg
Apple Elstar - Flickr - conall...jpg
Apple on desk.jpg
Apple on red background.jpg
Apple plate.jpg
Apple Slice - Flickr - places lost.jpg
Apples (2837340315).jpg
Apples (4362331143).jpg
Apples (6494047159).jpg
Apples - Flickr - munchflemming.jpg
Apples - Flickr - Southernpixel - Alby Headrick.jpg
181 365+1 Banana (7623220032).jpg
4 Bananes.jpg
A bunch of bananas.jpg
Baby bananas.jpg
Banana (2).jpg
Banana (2516487176).jpg
Banana - Q503.jpg
Banana 0757.jpg
Banana 10.jpg
Banana 5.jpg
Banana bonita.jpg
Banana bunch on wall.jpg
Banana Bunch.jpg
Banana caturra.JPG
Banana fruit (musa).jpg
Banana nanica.jpg
Banana part.jpg
Banana sadež.JPG
16-09-17-WikiLovesCocktails-Zutaten-Img0158.jpg
164 - oranges.jpg
2010-365-124 Still Life (4579895727).jpg
2014-365-111 Orange Space (13942821996).jpg
A Citrus fruit on a Table.jpg
A Green yellow Orange.jpg
A zoomed up picture of an orange.jpg
Ambersweet oranges.jpg
An Orange.jpg
An orange.jpg
Apelsinas.JPG
Apelsiner - Flickr - nilsw.jpg
Arancia.jpg
Arunachali Orange.jpg'''.splitlines()
WATER = '''A Glass Full of Water.jpg
AGUA (33456041042).jpg
Bengali ek glas pani.jpg
Blue tumbler.jpg
Bubbles in glass of water.jpg
Carafe d'eau.jpeg
Clean water (30313108598).jpg
CoCo Ichibanya Drinking water.jpg
Cold Water (117688999).jpeg
Cold water (1733066814).jpg
Cup With Water.jpg
DicWater.jpg
Drei Gläser.jpg
Drinkwater.jpg
Duritky.jpg
Edalontzia.jpg
Fasting 4-Fasting-a-glass-of-water-on-an-empty-plate.jpg
Flickr - cyclonebill - Vand (6).jpg
Glas halbvoll.JPG
Glass Half Full bw 1.JPG
Glass half full or half empty.png
Glass Half Full.jpg
Glass of cold mineral water.jpg
Glass of Water (50838445027).jpg
Glass of Water - Flickr - Greg Riegler Photography.jpg
Glass of water on vintage sheet.jpg
Glass of water with ice cubes.JPG
Glass of water, detail.jpg
Glass of Water.JPG
Glass-half-full.jpeg
Glass-of-water.jpg
Glasses of water -- Candolim.jpg
Glasses of water.jpg
Ice water (11462178684).jpg
Ice Water (5685106294).jpg
Jamie Street 2016-06-13 (Unsplash).jpg
Kitchenware Glass Rezowan (4).JPG
Potiri.jpg
Punti di vista.png
Stilles Mineralwasser.jpg
Su bardağı Water Glass.jpg
Svatoanenská voda (H2O) 01.jpg'''.splitlines()
WATER += '''Vaso de agua.jpg
Water 2.jpg
Water 20211205 115735.jpg
Water at glass2.jpg
Water glass9.jpg
Water pour 2 (59785756).jpg
Water with bokeh (3654022771).jpg
Water7.jpg
飲水 (4820799273).jpg'''.splitlines()

def clean(value):
    return html.unescape(re.sub('<[^>]+>', '', str(value))).strip()

def get(url):
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={'User-Agent': UA}), timeout=45) as f:
                return f.read()
        except HTTPError as exc:
            # Preserve rate-limit failures for audit instead of repeatedly retrying.
            if exc.code == 429 or attempt == 2:
                raise
            time.sleep(1 + attempt)
        except Exception:
            if attempt == 2: raise
            time.sleep(1 + attempt)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def fetch_metadata(only_titles=None, cache_prefix='batch', write_candidates=True):
    records = []
    lookup = {('File:' + n).replace('_',' '):c for c,names in [('food',FOOD),('water',WATER)] for n in names}
    titles = only_titles or list(lookup)
    for start in range(0, len(titles), 10):
        params = {'action':'query','format':'json','prop':'imageinfo','titles':'|'.join(titles[start:start+10]),
                  'iiprop':'url|extmetadata|sha1|size|mime', 'iiurlwidth':512}
        url = 'https://en.wikipedia.org/w/api.php?' + urlencode(params)
        blob = get(url)
        (DATA/'metadata'/f'{cache_prefix}_{start:03d}.json').write_bytes(blob)
        for page in json.loads(blob)['query']['pages'].values():
            if not page.get('imageinfo'): continue
            info = page['imageinfo'][0]
            meta = info.get('extmetadata',{})
            val = lambda k: clean(meta.get(k,{}).get('value',''))
            title = page['title']
            license_name = val('LicenseShortName')
            # Commons permits other licenses, but this first set deliberately accepts
            # only explicit Creative Commons or public-domain statements.
            allowed = license_name.startswith('CC ') or license_name == 'CC0' or 'public domain' in license_name.lower()
            rid = sha(title.encode())[:12]
            records.append({'id':rid,'title':title,'category':lookup[title],
                'source_url':info['descriptionurl'],'original_url':info['url'],
                'download_url':info.get('thumburl',info['url']), 'original_sha1':info['sha1'],
                'original_width':info['width'],'original_height':info['height'],
                'author':val('Artist'),'description':val('ImageDescription'),
                'license':license_name,'license_url':val('LicenseUrl'),
                'attribution_required':val('AttributionRequired'),'credit':val('Credit'),
                'commons_categories':val('Categories'),'allowed_license':allowed,
                'metadata_api_url':url,'path':f'data/images/{rid}.jpg'})
        print('metadata', min(start+10,len(titles)),len(titles),flush=True)
    if write_candidates:
        (DATA/'candidates.json').write_text(json.dumps({'images':records},ensure_ascii=False,indent=2))
    return records

def download_one(row):
    if not row['allowed_license']: return {**row,'download_status':'excluded_license'}
    path=ROOT/row['path']
    try:
        raw=get(row['download_url'])
        from io import BytesIO
        im=ImageOps.exif_transpose(Image.open(BytesIO(raw))).convert('RGB')
        im.thumbnail((640,640),Image.Resampling.LANCZOS)
        im.save(path,format='JPEG',quality=95)
        return {**row,'download_status':'ok','downloaded_sha256':sha(raw),
                'sha256':sha(path.read_bytes()),'width':im.width,'height':im.height,
                'modification':'EXIF orientation applied; RGB; max dimension 640; JPEG quality 95; metadata omitted'}
    except Exception as e:
        return {**row,'download_status':'error','error':str(e)}

def contact_sheets(rows):
    for category in ['food','water']:
        subset=[r for r in rows if r['category']==category and r['download_status']=='ok']
        for page,start in enumerate(range(0,len(subset),24)):
            sheet=Image.new('RGB',(1200,1104),'white');draw=ImageDraw.Draw(sheet)
            for i,r in enumerate(subset[start:start+24]):
                im=Image.open(ROOT/r['path']);im.thumbnail((184,230))
                x=(i%6)*200;y=(i//6)*276
                sheet.paste(im,(x+(192-im.width)//2,y))
                draw.text((x+4,y+234),r['id'],fill='black')
                draw.text((x+4,y+248),r['title'][5:30],fill='black')
            sheet.save(DATA/f'contact_{category}_{page+1}.jpg')

def finalize(selection):
    allrows=json.loads((DATA/'candidates_downloaded.json').read_text())['images']
    selected=json.loads(Path(selection).read_text())
    keep={rid:split for split,ids in selected['splits'].items() for rid in ids}
    rows=[]
    for row in allrows:
        if row['id'] in keep:
            if row['download_status']!='ok': raise ValueError(row)
            rows.append({**row,'split':keep[row['id']],'visual_review':'accepted: photograph; resource visible; no obvious legible text/watermark at model resolution'})
    if len(rows)!=len(keep): raise ValueError('missing or duplicate selection ID')
    if len({r['original_sha1'] for r in rows}) != len(rows): raise ValueError('duplicate original SHA1')
    for category in ['food','water']:
        print(category, {split:sum(r['category']==category and r['split']==split for r in rows) for split in ['train','test']})
    manifest={'dataset_name':'commons_resource_photos_v1','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'purpose':'local visual emergent communication stimuli; labels never passed to policy',
              'split_unit':'Commons original image; related shots manually checked during selection',
              'selection_notes':selected.get('notes',[]),'images':rows}
    (DATA/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    attrib=['# Photo credits','', 'Images retain their individual licenses. Local copies were resized and converted to RGB JPEG; no semantic image edits were made.','']
    for r in rows:
        attrib.append(f"- `{r['id']}` — [{r['title']}]({r['source_url']}), {r['author']}; [{r['license']}]({r['license_url'] or r['source_url']}); {r['split']}.")
    (DATA/'ATTRIBUTION.md').write_text('\n'.join(attrib)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--finalize');p.add_argument('--skip-metadata',action='store_true');a=p.parse_args()
    for d in [DATA,DATA/'images',DATA/'metadata']: d.mkdir(parents=True,exist_ok=True)
    if a.finalize: finalize(a.finalize);return
    rows=json.loads((DATA/'candidates.json').read_text())['images'] if a.skip_metadata else fetch_metadata()
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(download_one,rows):
            results.append(row);print(row['category'],row['id'],row['download_status'],flush=True)
    (DATA/'candidates_downloaded.json').write_text(json.dumps({'images':results},ensure_ascii=False,indent=2))
    contact_sheets(results)

if __name__=='__main__': main()
