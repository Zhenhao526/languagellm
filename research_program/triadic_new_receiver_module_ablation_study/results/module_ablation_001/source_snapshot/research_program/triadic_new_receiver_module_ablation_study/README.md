# New-receiver module ablation

This package complements the frozen-pair transmission run by training only
one class of new C modules at a time.  The action-only and sender-only arms
are paired with the audited full-C transmission result.  A/B remain frozen.

The task uses local NumPy float64 tanh MLPs and the same strict altpair PL
environment; no LLM, pretrained language model, vision model or network
service is used.

Run the static preparation, execution, independent audit and JSON summary with
the runner, audit and summarize modules in this directory.  The experiment
uses eight source seeds, two schedules, two selective arms and live/silent
routing.
