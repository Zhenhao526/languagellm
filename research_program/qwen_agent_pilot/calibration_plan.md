# Matched channel calibration protocol

Protocol frozen before collection on 2026-09-20. This is a small screening experiment, not a confirmatory study and not a claim that a language has emerged.

## Question

Does free symbol communication improve coordinated collection over an empty channel when model, task, roles, feedback, exposures, decoding settings and paired episode schedule are held fixed? Can the model solve the same task when sender and receivers are given a complete shared codebook?

## Conditions

- **Blank:** requester receives the private order but the channel is closed. Helpers receive an empty message. The requester still takes a model turn, so each episode retains the same three-call structure.
- **Known codebook:** requester and helpers receive the same fixed mapping between six arbitrary six-character strings and complete orders. This is a positive control for task execution with successful decoding supplied.
- **Free symbols:** current symbol-only protocol; the requester sends one 4–8 character string from `@#%&+=?~`; helpers receive no mapping.

The codebook is fixed across seeds and has no message shorter than six characters. It is a task-competence check, not a candidate emergence condition. The free-symbol arm uses no natural-language channel. Each agent retains a separate context; action selection remains simultaneous; all conditions receive the same scalar team feedback and observe the same actions.

## Matched schedule and model settings

For each seed, the same 36 episodes are replayed in each condition: six complete orders crossed with all three requester identities, repeated in two shuffled blocks. The three seeds are `20260921`, `20260922`, and `20260923`. The message-generation and helper-call seeds are paired across conditions. Condition order rotates across paired seeds. Each condition starts with fresh agent histories. Model revision is `16daa4818c54ce5f5436f929d52542eb65bbed9d`; temperature is 0.35 and maximum completion length is 120 tokens. Raw completions and hidden reasoning are not retained.

## Outcomes

Report seed-level and pooled rates for joint success, designated item accuracy, destination accuracy, designated helper's combined correctness, and unassigned-helper waiting. For symbol conditions, report protocol validity, distinct strings, exact repeats within requester–meaning pairs, cross-requester agreement by meaning, and empirical mutual information of message with requester and meaning. For the known-codebook condition, additionally report exact encoder accuracy. Blank-channel message metrics are undefined by design.

## Interpretation rules

This 3-seed, 36-episode-per-cell screen has limited precision. Do not use a significance claim or generalize to human language origins. Low known-codebook performance means task/prompt competence must be improved before interpreting free-symbol failure. Strong codebook performance with weak free-symbol performance isolates a communication/convention bottleneck more clearly, but does not distinguish exploration, feedback credit assignment, or insufficient exposure. Strong free-symbol performance requires follow-up transfer and held-out meaning tests before calling it a stable convention.

## Resume and records

The runner writes an atomic checkpoint after every seed-condition run. The result contains episode-level goals, boards, messages, actions, outcomes, source settings, and per-cell summaries, but no raw completions. A resumed run must use exactly the checkpoint's seed list, model, temperature and token limit.
