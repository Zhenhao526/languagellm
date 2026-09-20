# Local storage sweep and repository sync

- Repository: `git@github.com:Zhenhao526/languagellm.git`, branch `main`.
- Before this receipt was added, local `main` matched `origin/main` at `e78b4baa707e67402f5e2276f400fe1e690ac04e`; the source, research history, reports, compact results, and audit receipts already present locally were therefore synced.
- Removed only rebuildable caches: Homebrew's unused download cache (about 1.6 GB reclaimed), pip's cache (about 251 MB reclaimed), and generated Python bytecode directories.
- Kept all research sources, formal results, reports, Git history, the `.exp_venv` used by several experiments, and the active Qwen model and matrix checkpoint file.
- Left the 901 MB staged VS Code update untouched because its updater was active.
- The v3 Qwen matrix was still running during this sweep; its partial checkpoint was not treated as a final archive and was left intact for the run to resume/complete.
- See `LOCAL_CLEANUP_RECEIPT_2026-09-20_storage_sweep.json` for the detailed cleanup record.
