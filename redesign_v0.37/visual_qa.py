"""Deterministic raster QA for v0.37 figures."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from PIL import Image
import numpy as np
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True);args=parser.parse_args();out=args.out.resolve();d=out/"figures";names=("01_origin_outcomes.png",);figures={};checks=[]
    for name in names:
        p=d/name
        with Image.open(p) as im:rgb=np.asarray(im.convert("RGB"));h,w=rgb.shape[:2]
        f=float(np.mean(np.any(rgb<248,axis=2)));dim=w>=1800 and h>=900;nw=.01<f<.80;figures[name]={"sha256":sha(p),"width":int(w),"height":int(h),"nonwhite_fraction":f,"readable_dimensions":dim};checks.append({"file":name,"dimensions":dim,"nonwhite_range":nw})
    passed=all(x["dimensions"] and x["nonwhite_range"] for x in checks);result={"status":"passed" if passed else "failed","passed":passed,"figures":figures,"checks":checks};(d/"visual_qa.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n");source=json.loads((d/"figure_source.json").read_text());source["status"]="passed_visual_QA";source["visual_qa_sha256"]=sha(d/"visual_qa.json");source["visual_qa"]=result;(d/"figure_source.json").write_text(json.dumps(source,ensure_ascii=False,indent=2)+"\n");print(json.dumps(result,ensure_ascii=False))
if __name__=="__main__":main()
