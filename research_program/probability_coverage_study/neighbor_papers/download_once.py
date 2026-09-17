"""Download exactly one named primary paper; refuse a second attempt.

No retry logic. Redirects are followed and the final URL is recorded.
Run with the bundled document Python (pypdf), not a model runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
SOURCES = {
    "lu": {
        "year": 2020,
        "author": "Lu",
        "title": "Countering Language Drift with Seeded Iterated Learning",
        "url": "https://proceedings.mlr.press/v119/lu20c/lu20c.pdf",
        "filename": "2020_Lu_Countering Language Drift with Seeded Iterated Learning.pdf",
    },
    "korbak": {
        "year": 2019,
        "author": "Korbak",
        "title": "Developmentally motivated emergence of compositional communication via template transfer",
        "url": "https://arxiv.org/pdf/1910.06079",
        "filename": "2019_Korbak_Developmentally motivated emergence of compositional communication via template transfer.pdf",
    },
    "lazaridou": {
        "year": 2020,
        "author": "Lazaridou",
        "title": "Multi-agent Communication meets Natural Language: Synergies between Functional and Structural Language Learning",
        "url": "https://aclanthology.org/2020.acl-main.685.pdf",
        "filename": "2020_Lazaridou_Multi-agent Communication meets Natural Language_Synergies between Functional and Structural Language Learning.pdf",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper", choices=tuple(SOURCES))
    args = parser.parse_args()
    source = SOURCES[args.paper]
    receipt = ROOT / f"{args.paper}_download.json"
    target = ROOT / source["filename"]
    raw = ROOT / f"{args.paper}_response.unverified"
    # The exclusive initial receipt makes failed and interrupted attempts final.
    record = {
        "paper_id": args.paper,
        **source,
        "requested_url": source["url"],
        "actual_url": None,
        "started_at_utc": utc_now(),
        "attempts": 1,
        "automatic_retries": 0,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": "attempt_started",
    }
    if target.exists() or raw.exists():
        raise FileExistsError("A response/target already exists; no new attempt allowed")
    with receipt.open("x", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
        f.write("\n")
    try:
        request = urllib.request.Request(source["url"], headers={"User-Agent": "ResearchPaperArchive/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
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
        raw.write_bytes(content)
        record["bytes"] = len(content)
        record["sha256"] = hashlib.sha256(content).hexdigest()
        record["pdf_header_verified"] = content.startswith(b"%PDF-")
        if not record["pdf_header_verified"]:
            raise ValueError("Response is not a PDF header")
        reader = PdfReader(io.BytesIO(content))
        page_texts = [page.extract_text() or "" for page in reader.pages]
        record["pages"] = len(page_texts)
        record["all_pages_text_extracted"] = True
        record["total_extracted_characters"] = sum(map(len, page_texts))
        record["first_page_excerpt"] = page_texts[0][:1800]
        record["title_verified"] = normalized(source["title"]) in normalized(page_texts[0])
        record["author_verified"] = normalized(source["author"]) in normalized(page_texts[0])
        if not record["title_verified"] or not record["author_verified"]:
            raise ValueError("First-page title/author verification failed")
        raw.rename(target)
        record["saved_path"] = str(target)
        record["status"] = "downloaded_and_verified"
    except Exception as error:
        record["status"] = "failed_no_retry"
        record["error_type"] = type(error).__name__
        record["error"] = str(error)
        if raw.exists():
            record["unverified_response_path"] = str(raw)
    finally:
        record["completed_at_utc"] = utc_now()
        receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: record.get(k) for k in ["paper_id", "status", "actual_url", "bytes", "sha256", "pages", "title_verified", "error"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
