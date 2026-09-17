"""Bounded downloads of the six primary papers selected by the online search."""
from pathlib import Path
import hashlib,json,time,urllib.request,urllib.error,datetime
from pypdf import PdfReader
HERE=Path(__file__).resolve().parent
TARGETS=[
 ('luna2020','2020_RodriguezLuna_Object_Constancy_FindingsEMNLP','https://aclanthology.org/2020.findings-emnlp.397/','https://aclanthology.org/2020.findings-emnlp.397.pdf'),
 ('garcia2022','2022_Garcia_Disentangling_Categorization_NAACL','https://aclanthology.org/2022.naacl-main.335/','https://aclanthology.org/2022.naacl-main.335.pdf'),
 ('slowik2021','2021_Slowik_Structural_Inductive_Biases_arXiv2002.01335v4','https://arxiv.org/abs/2002.01335','https://arxiv.org/pdf/2002.01335v4'),
 ('ri2023','2023_Ri_Emergent_Communication_with_Attention_arXiv2305.10920v1','https://arxiv.org/abs/2305.10920','https://arxiv.org/pdf/2305.10920v1'),
 ('kouwenhoven2024','2024_Kouwenhoven_Representational_Alignment_CMCL','https://aclanthology.org/2024.cmcl-1.5/','https://aclanthology.org/2024.cmcl-1.5.pdf'),
 ('imai2024','2024_Imai_Pretrained_Generative_Models_Reassembly_TJSAI','https://www.jstage.jst.go.jp/article/tjsai/39/2/39_39-2_D-N71/_article/-char/en','https://www.jstage.jst.go.jp/article/tjsai/39/2/39_39-2_D-N71/_pdf')]

def main():
    entries=[]
    for key,name,page,url in TARGETS:
        row=dict(id=key,basename=name,official_page=page,pdf_url=url,
                 requested_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        path=HERE/'primary'/(name+'.pdf')
        try:
            if not path.exists():
                req=urllib.request.Request(url,headers={'User-Agent':'ResearchLiteratureAudit/1.0 (six-paper method comparison; no bulk crawl)'})
                with urllib.request.urlopen(req,timeout=40) as response:
                    raw=response.read();row['http_status']=response.status;row['final_url']=response.url
                    row['headers']=dict(response.headers)
                if not raw.startswith(b'%PDF'):
                    (HERE/'logs'/(key+'_non_pdf.response')).write_bytes(raw)
                    raise ValueError('Response is not a PDF; preserved, no alternate entry used.')
                path.write_bytes(raw)
            reader=PdfReader(path)
            text='\n\n'.join(f'=== PDF PAGE {i} ===\n'+(p.extract_text() or '') for i,p in enumerate(reader.pages,1))
            (HERE/'primary'/(name+'.pages.txt')).write_text(text)
            row.update(status='downloaded_and_extracted',pdf_file=str(path.relative_to(HERE)),
                       pdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pages=len(reader.pages),text_characters=len(text))
        except Exception as exc:
            row.update(status='failed',error=repr(exc))
            if isinstance(exc,urllib.error.HTTPError):
                row['http_status']=exc.code;row['headers']=dict(exc.headers)
                (HERE/'logs'/(key+'_error.response')).write_bytes(exc.read())
        entries.append(row)
        (HERE/'primary'/'manifest.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2)+'\n')
        print(key,row['status'],row.get('pages'),flush=True)
        time.sleep(2)

if __name__=='__main__':main()
