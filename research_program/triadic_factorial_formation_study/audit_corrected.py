"""Independent replay audit for ``correction_001``.

The frozen audit module is retained verbatim for provenance.  Its one
researcher-side factor subset function is replaced here with the corrected
elementwise implementation before replay; all neural forwards, settlements,
hash checks and paired-log checks remain the frozen audit procedure.
"""
import sys
import platform
from pathlib import Path
import numpy as np

from . import audit as legacy
from . import correction
from research_program.triadic_action_dependency_study import runner as old


def main():
    legacy.platform = platform
    legacy.array_sha = old.array_sha
    legacy.core = old.core

    def corrected_factor_subset(case_spec, values, edge_indices):
        return correction.corrected_subset(case_spec, values, edge_indices)

    legacy.factor_subset_independent = corrected_factor_subset

    # The corrected run tree reuses immutable NPZ/checkpoint files through
    # absolute symlinks.  Treat equivalent resolved paths as identical while
    # retaining the frozen recursive comparison for every other field.
    original_compare = legacy.compare

    def compare_with_resolved_paths(left, right, label='value', tolerance=2e-12):
        if isinstance(left, str) and isinstance(right, str):
            try:
                if Path(left).exists() and Path(right).exists() and Path(left).resolve() == Path(right).resolve():
                    return
            except OSError:
                pass
        return original_compare(left, right, label, tolerance)

    legacy.compare = compare_with_resolved_paths
    legacy.main()


if __name__ == '__main__':
    main()
