"""Explicitly authorized one-time Lu retry; preserve the first failed receipt."""

import hashlib
import io
import json
import signal
import urllib.error
import urllib.request
from pathlib import Path

from pypdf import PdfReader
from download_once import ROOT, SOURCES, normalized, utc_now


def timed_out(_signum, _frame):
    raise TimeoutError("Authorized retry exceeded the 30-second network deadline")


def main():
    source = SOURCES["lu"]
    original = ROOT / "lu_download.json"
    old = json.loads(original.read_text())
    assert old["status"] == "failed_no_retry"
    receipt = ROOT / "lu_download_retry_001.json"
    target = ROOT / source["filename"]
    raw = ROOT / "lu_retry_001_response.unverified"
    if target.exists() or raw.exists():
        raise FileExistsError("Existing retry response; do not issue another request")
    record = {
        "paper_id": "lu",
        **source,
        "requested_url": source["url"],
        "actual_url": None,
        "started_at_utc": utc_now(),
        "attempts_this_receipt": 1,
        "overall_attempt_ordinal": 2,
        "automatic_retries": 0,
        "network_deadline_seconds": 30,
        "authorization": "Root explicitly authorized one same-URL retry after the preserved DNS failure; no third attempt.",
        "first_failure_receipt": str(original),
        "first_failure_receipt_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "download_helper_sha256": hashlib.sha256((ROOT / "download_once.py").read_bytes()).hexdigest(),
        "status": "attempt_started",
    }
    with receipt.open("x", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
        f.write("\n")
    try:
        signal.signal(signal.SIGALRM, timed_out)
        signal.alarm(30)
        try:
            request = urllib.request.Request(source["url"], headers={"User-Agent": "ResearchPaperArchive/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                record["actual_url"] = response.geturl()
                record["http_status"] = response.status
                record["content_type"] = response.headers.get("Content-Type")
                content = response.read()
        except urllib.error.HTTPError as error:
            record["actual_url"] = error.geturl()
            record["http_status"] = error.code
            record["content_type"] = error.headers.get("Content-Type")
            content = error.read()
            raw.write_bytes(content)
            record["bytes"] = len(content)
            record["sha256"] = hashlib.sha256(content).hexdigest()
            raise
        finally:
            signal.alarm(0)
        raw.write_bytes(content)
        record["bytes"] = len(content)
        record["sha256"] = hashlib.sha256(content).hexdigest()
        record["pdf_header_verified"] = content.startswith(b"%PDF-")
        if not record["pdf_header_verified"]:
            raise ValueError("Response is not a PDF header")
        reader = PdfReader(io.BytesIO(content))
        texts = [page.extract_text() or "" for page in reader.pages]
        record["pages"] = len(texts)
        record["all_pages_text_extracted"] = True
        record["total_extracted_characters"] = sum(map(len, texts))
        record["first_page_excerpt"] = texts[0][:1800]
        record["title_verified"] = normalized(source["title"]) in normalized(texts[0])
        record["author_verified"] = normalized(source["author"]) in normalized(texts[0])
        if not record["title_verified"] or not record["author_verified"]:
            raise ValueError("Title/author verification failed")
        raw.rename(target)
        record["saved_path"] = str(target)
        record["status"] = "downloaded_and_verified"
    except Exception as error:
        record["status"] = "failed_stop_no_further_attempt"
        record["error_type"] = type(error).__name__
        record["error"] = str(error)
        if raw.exists():
            record["unverified_response_path"] = str(raw)
    finally:
        record["completed_at_utc"] = utc_now()
        receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: record.get(k) for k in ["status", "actual_url", "bytes", "sha256", "pages", "title_verified", "error"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
