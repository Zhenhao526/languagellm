from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,urllib.request
root=Path(__file__).resolve().parent
items=[('2004_Greensmith_Bartlett_Baxter_Variance_Reduction_Techniques.pdf','https://www.jmlr.org/papers/volume5/greensmith04a/greensmith04a.pdf'),('2018_Wu_et_al_Action_Dependent_Factorized_Baselines.pdf','https://arxiv.org/pdf/1803.07246')]
records=[]
for name,url in items:
 dest=root/name;assert not dest.exists()
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'LocalAcademicResearch/1.0'}),timeout=60) as r:
  content=r.read();rec={'name':name,'url':url,'resolved_url':r.url,'utc':datetime.now(timezone.utc).isoformat(),'status':r.status,'content_type':r.headers.get('Content-Type'),'bytes':len(content)}
 if rec['status']==200 and content.startswith(b'%PDF'):
  dest.write_bytes(content);rec['sha256']=hashlib.sha256(content).hexdigest();rec['downloaded']=True
 else:
  (root/(name+'.response')).write_bytes(content);rec['downloaded']=False
 records.append(rec);(root/'download_receipt.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
