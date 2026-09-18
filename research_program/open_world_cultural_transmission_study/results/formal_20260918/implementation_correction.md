# Implementation correction

A preliminary diagnostic labeled `expanded_single_none` but still ran the standard update loop, making it equivalent to a fresh–fresh training arm. That batch was recognized before formal analysis, deleted, and excluded. The formal rerun sets `updates=0` for every `expanded_single_none` condition; its checkpoints contain only the initialized replacement worker and are audited as such.
