# Incumbent-worker surface transfer study

This targeted follow-up asks whether a learned convention is attached to an
agent's internal protocol or to the visible object labels used by its partner.
It continues the binary object-surface remapping study while making the
adaptation intervention explicit.

The parent has four rotating partners. A two-slot `dual2` sender learns which
two object types are needed at two successive sites. The sender is frozen after
3,000 updates. For each seed, an `incumbent` child copies worker 0 from the
parent, while a `fresh` child gets an independently initialized worker. Both
children then face either the original identity mapping or a stable swap of
the visible object labels. Live and silent channels are paired controls.

The primary readouts are held-out full-support return at initialization, final
return after 3,000 child updates, and the repair gain. The all-goal
`natural−permuted` check verifies that messages affect behavior. Nine seeds
yield 9 parent and 72 child runs. The experiment does not claim human language;
it measures protocol grounding, transfer cost, and reward-based repair in a
fully auditable tabular environment.

Run the source test with:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .exp_venv/bin/python \
  research_program/object_surface_transfer_study/tests/test_game.py
```
