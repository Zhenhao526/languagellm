# Frozen-pair new-receiver transmission

This package runs the next experiment in the language-emergence program.  It
freezes two agents from the completed factorized PL coordination study,
replaces the third with a fresh policy, and compares adaptation with live
versus silent packet routing.

The package is self-contained and freezes hashes of the source code and source
checkpoints before training.  The source and the new policy are local NumPy
float64 tanh MLPs.  No LLM, pretrained language model, vision model, or
network service is used.

Typical execution:

    /Users/xia/Documents/ChatGPT/语言/.venv/bin/python -m \
    research_program.triadic_new_receiver_transmission_study.runner prepare \
    --out research_program/triadic_new_receiver_transmission_study/results/transmission_001

After execution, replay the checkpoints and aggregate JSON with the audit and
summarize modules in this package.  Plotting uses plot_results.py.
