# Matched full-combination receiver control

This is a post-hoc frozen-policy comparison.  The seen-joint-combination A
policy comes from `holdout_001`; the matched full-combination A policy comes
from generation-2 `replace_A` in `chain_001`.  The two source families use the
same parent B/C endpoint, the same fresh A initialization, and the same
world/message/rematching streams.  Only A's training support differs.

Both policies are probed on the same heldout-combination aligned/placebo
cases.  No new optimizer updates are made.
