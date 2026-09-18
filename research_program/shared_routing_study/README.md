# Shared routing and role symmetry

This study asks whether a sender and a receiver can discover a reusable
compositional coordinate when they share the routing map that assigns message
slots to latent attributes. It reuses the four-object, two-attribute
referential game from `routed_factorization_study`.

The matched arms are `holistic`, fixed `factorized`, independently learned
`routed`, and `tied_routed`. The first three are the existing controls. The
new arm keeps separate sender and receiver factor tables, but ties their
row-softmax slot-to-attribute routing matrix. Thus tying supplies a shared
coordinate without supplying the token meanings or the factor tables.

The primary endpoint is held-out-combination return after fresh-agent
adaptation. Held-out-value support is a strict negative control: no arm can
recover an unseen atomic value from a routing constraint alone. Message
permutation, silent-channel, cross-partner consistency, route alignment and
recombined-message return are secondary endpoints. The formal grid retains
aligned/conflict communities, hidden/visible partner identity and full,
held-out-combination and held-out-value support so the role-sharing effect is
separated from surface conflict and partner-specific coding.

This is an architectural mechanism test. A positive `tied_routed` result
would show that a shared compositional coordinate is sufficient for transfer;
it would not show that an unconstrained population invented language.

Raw training logs and checkpoints are temporary. The archive keeps the frozen
source snapshot, compact endpoints, independent replay audit and cleanup
receipt.

## Reproduce

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.exp_venv/bin/python -m research_program.shared_routing_study.runner prepare --out /tmp/shared_routing_prepared
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.exp_venv/bin/python -m research_program.shared_routing_study.runner execute --prepared /tmp/shared_routing_prepared --out /tmp/shared_routing_execution --updates 3000
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.exp_venv/bin/python -m research_program.shared_routing_study.audit --prepared /tmp/shared_routing_prepared --execution /tmp/shared_routing_execution/execution --out /tmp/shared_routing_audit.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.exp_venv/bin/python -m research_program.shared_routing_study.aggregate --results /tmp/shared_routing_execution/execution/results.json --out /tmp/shared_routing_analysis.json --markdown /tmp/shared_routing_analysis.md
```
