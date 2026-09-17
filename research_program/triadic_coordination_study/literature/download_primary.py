"""One bounded attempt per observed primary-source PDF; retain failures."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import io
import json
import ssl
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
PAPERS = [
    ("2016_Wei_Luke_Lenient_Learning_in_Independent_Learner_Stochastic_Cooperative_Games_JMLR.pdf",
     "https://www.jmlr.org/papers/volume17/15-417/15-417.pdf", 42,
     ["Lenient Learning", "Ermo Wei", "Sean Luke"]),
    ("2019_Mahajan_MAVEN_Multi_Agent_Variational_Exploration_NeurIPS.pdf",
     "https://proceedings.neurips.cc/paper_files/paper/2019/file/f816dc0acface7498e10496222e9db10-Paper.pdf", 12,
     ["MAVEN", "Mahajan", "Whiteson"]),
    ("2022_Strupl_Reward_Weighted_Regression_Converges_to_a_Global_Optimum_AAAI.pdf",
     "https://ojs.aaai.org/index.php/AAAI/article/view/20811/20570", 9,
     ["Reward-Weighted Regression", "Faccio", "Schmidhuber"]),
    ("2023_Zhao_Local_Optimization_Achieves_Global_Optimality_in_Multi_Agent_Reinforcement_Learning_ICML.pdf",
     "https://proceedings.mlr.press/v202/zhao23j/zhao23j.pdf", 27,
     ["Local Optimization Achieves Global Optimality", "Yulai Zhao", "Jason"]),
]


def main():
    out = HERE / "download_001"
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, url, expected_pages, checks in PAPERS:
        row = {"filename": name, "url": url, "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "attempts": 1, "expected_pages": expected_pages, "identity_checks": checks}
        start = time.monotonic()
        try:
            request = Request(url, headers={"User-Agent": "LanguageEmergenceResearch/0.1 (literature retrieval)"})
            with urlopen(request, timeout=30, context=ssl.create_default_context()) as response:
                row.update(http_status=response.status, final_url=response.url, headers=dict(response.headers))
                body = response.read(16 * 1024 * 1024 + 1)
            row.update(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
            if len(body) > 16 * 1024 * 1024 or not body.startswith(b"%PDF-"):
                raise ValueError("Oversize or non-PDF response")
            reader = PdfReader(io.BytesIO(body))
            page_texts = [page.extract_text() or "" for page in reader.pages]
            first = page_texts[0]
            compact = " ".join(first.split())
            if len(reader.pages) != expected_pages or not all(c.casefold() in compact.casefold() for c in checks):
                raise ValueError("PDF page count or first-page identity mismatch")
            (out / name).write_bytes(body)
            text_path = out / name.replace(".pdf", ".pages.txt")
            text_path.write_text("\n\n".join(f"=== PDF PAGE {i+1} ===\n{t}" for i, t in enumerate(page_texts)))
            row.update(status="downloaded_and_identity_checked", pages=len(reader.pages),
                       text_sha256=hashlib.sha256(text_path.read_bytes()).hexdigest())
        except HTTPError as error:
            body = error.read(8192)
            error_name = name.replace(".pdf", ".http_error")
            (out / error_name).write_bytes(body)
            row.update(status="failed", http_status=error.code, error=str(error),
                       headers=dict(error.headers), error_body=error_name,
                       error_sha256=hashlib.sha256(body).hexdigest(), bytes=len(body))
        except Exception as error:
            row.update(status="failed", error_type=type(error).__name__, error=str(error))
        row["elapsed_seconds"] = time.monotonic() - start
        rows.append(row)
        with (out / "requests.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({k: row.get(k) for k in ("filename", "status", "pages", "bytes", "error")}), flush=True)
    receipt = {"status": "completed_attempts", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "rows": rows, "retry_count": 0, "pretrained_model_calls": 0}
    (out / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
