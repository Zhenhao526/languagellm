"""Archive one already-identified primary paper; no automatic retries.

Reuses the prior archival routine without changing its source or library.
The routine records final URL, time, bytes, SHA256, PDF pages and title/author.
"""
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "probability_coverage_study/neighbor_papers/download_once.py"
spec = importlib.util.spec_from_file_location("primary_archive_routine", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.ROOT = HERE / "papers"
module.ROOT.mkdir(exist_ok=True)
module.SOURCES = {
    "evtimova": {
        "year": 2018,
        "author": "Evtimova",
        "title": "Emergent Communication in a Multi-Modal, Multi-Step Referential Game",
        "url": "https://arxiv.org/pdf/1705.10369",
        "filename": "2018_Evtimova_Emergent Communication in a Multi-Modal Multi-Step Referential Game.pdf",
    },
    "kalinowska": {
        "year": 2022,
        "author": "Kalinowska",
        "title": "Towards Situated Communication in Multi-Step Interactions: Time is a Key Pressure in Communication Emergence",
        "url": "https://olakali.github.io/assets/pdfs/2022CogSciKaDaMaMuPi.pdf",
        "filename": "2022_Kalinowska_Towards Situated Communication in Multi-Step Interactions_Time is a Key Pressure in Communication Emergence.pdf",
    },
    "fay": {
        "year": 2018,
        "author": "Fay",
        "title": "How to Create Shared Symbols",
        "url": "https://eprints.gla.ac.uk/157897/7/157897.pdf",
        "filename": "2018_Fay_How to Create Shared Symbols_Accepted Version.pdf",
    },
}

if __name__ == "__main__":
    module.main()
