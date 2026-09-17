# Single-slot compositional transfer probe

This package is a post-hoc frozen-policy probe built on the final matched
full-combination control.  It tests whether heldout aligned-minus-placebo
transfer is localized to one first-window message slot.  Four single-slot
masks and a full-packet reference are evaluated on the same 3,456 heldout
cases for each of 64 source policies.  Silent-channel rows are exact aliases;
no optimizer updates are made.

The final formal output is written under `results/slot_transfer_004/`, with an
independent replay under `audit_slot_transfer_004/` and aggregate files under
`results/summary_slot_transfer_004/`. Earlier numbered folders are retained as
development batches; `slot_transfer_004` is the authority used in the report.
