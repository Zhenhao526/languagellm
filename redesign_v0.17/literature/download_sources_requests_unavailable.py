from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,requests
root=Path(__file__).resolve().parent
items=[('2004_Greensmith_Bartlett_Baxter_Variance_Reduction_Techniques.pdf','https://www.jmlr.org/papers/volume5/greensmith04a/greensmith04a.pdf'),('2018_Wu_et_al_Action_Dependent_Factorized_Baselines.pdf','https://arxiv.org/pdf/1803.07246')]
records=[]
for name,url in items:
 dest=root/name;assert not dest.exists()
 r=requests.get(url,timeout=60,headers={'User-Agent':'LocalAcademicResearch/1.0'})
 rec={'name':name,'url':url,'resolved_url':r.url,'utc':datetime.now(timezone.utc).isoformat(),'status':r.status_code,'content_type':r.headers.get('Content-Type'),'bytes':len(r.content)}
 if r.status_code==200 and r.content.startswith(b'%PDF'):
  dest.write_bytes(r.content);rec['sha256']=hashlib.sha256(r.content).hexdigest();rec['downloaded']=True
 else:
  (root/(name+'.response')).write_bytes(r.content);rec['downloaded']=False
 records.append(rec);(root/'download_receipt.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
