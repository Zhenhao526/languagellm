# Two-generation transmission study

This study starts from the nine parent seeds whose staged dual2 protocol passed the frozen compositionality rule in the 32-seed parent batch. It executes two component substitutions in sequence.

- `worker_then_sender`: generation 1 replaces worker 0 with a slot-local receiver; generation 2 replaces the sender while that receiver is frozen.
- `sender_then_worker`: generation 1 replaces the sender; generation 2 replaces worker 0 with a slot-local receiver while the new sender is frozen.

Each generation has a paired `live` and `silent` adaptation arm. The stage-specific worlds, goals, partner identities and sampling uniforms are shared between the pair. The final endpoint is evaluated with natural messages, closed/silent messages, partner-permuted messages and slot-recombined messages. A composable endpoint has natural live return at least 0.60 and an absolute recombined-minus-natural gap at most 0.02.

The parent checkpoint tree and all per-update raw logs are temporary audit inputs. Only the frozen plan, compact results, aggregate analysis and audit receipt belong in the repository archive.
