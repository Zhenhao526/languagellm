# Analysis addendum

The frozen transfer analysis summarizes final endpoints and initial-to-final gains. A supplemental script, `supplemental_surface_initial_analysis.py`, was added after training to make the receiver-initialization × surface-mapping contrast explicit at the initial and final checkpoints. It reads the archived 288-row result table only. Its confidence intervals use paired Student t intervals (df=8); the primary endpoint and threshold are unchanged.

This addendum makes visible a contrast implicit in the frozen design: reverse−identity among incumbent receivers minus reverse−identity among fresh receivers. It is reported as a supplementary mechanism analysis and should be replicated in visual or learned-perception settings before generalizing beyond this tabular task.
