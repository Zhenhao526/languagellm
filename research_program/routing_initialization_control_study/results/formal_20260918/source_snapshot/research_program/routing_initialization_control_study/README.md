# Routing initialization control

This targeted control separates shared routing during learning from a merely
shared initialization. It compares `routed` (independent random routing),
`sync_routed` (independent routing matrices initialized identically and then
updated separately), and `tied_routed` (one shared routing parameter updated by
both roles). All arms retain separate sender and receiver factor tables.

The study uses the aligned, hidden-partner referential game and the full,
held-out-combination and held-out-value supports. The formal endpoint is
held-out-combination return under the same nine seeds and 3,000-update schedule
as the shared-routing matrix. If `tied_routed` remains above `sync_routed`,
the gain is evidence for learning-time role coupling rather than an artifact
of starting from the same route matrix.

This is a confound control for the shared-routing mechanism; it is not a claim
that any arm has invented human language.
