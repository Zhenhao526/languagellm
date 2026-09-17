# Two-generation protocol chain

This package follows the audited frozen-pair/new-receiver experiment with a
small cultural-transmission chain.  A fresh agent replaces A in generation 2,
then a fresh agent replaces B in generation 3.  Generation 3 inherits the
generation-2 live endpoint, so retention and routing effects can be measured
with fixed task mechanics.

Use `runner.py prepare`, `runner.py execute`, `audit.py`, `summarize.py` and
`plot_results.py` in that order.  The authoritative run is stored under
`results/chain_001` after preparation and execution.  Analysis scripts are
post-hoc and are not part of the frozen execution source manifest.
