# Three-context local Qwen pilot plan

## Question

Can three separately maintained contexts for one local Qwen model follow a symbol-only message constraint, use scalar action feedback, and begin to stabilize a shared code for object, attribute, responsible partner and destination?

## Pilot task

- Three agents (`A`, `B`, `C`) take turns as requester. The requester privately sees one collection order and broadcasts one message to the other two agents.
- Each order specifies one object type, one color attribute, one of the two helper agents responsible for collection, and one destination.
- Helpers see four candidate items, including one target and distractors. They do not see the private order. Both helpers independently return a structured action or wait; neither receives the other's response before acting.
- The episode succeeds only if the designated helper selects the exact object-attribute item, sends it to the named destination, and the other helper waits. Both helpers receive the same scalar outcome.
- Six meaning tuples are crossed with all three requester identities in each of two blocks (36 episodes). Thus every requester sends each meaning twice, once per block, separating sender identity from meaning and allowing one within-context repeat. This deliberately small support tests context separation and early convention maintenance; it does not test open-ended vocabulary.

## Communication constraint

Only `@#%&+=?~` strings of length 4–8 pass through the message channel. Invalid output becomes an empty message and is logged as a protocol violation; the runner does not repair it or expose the invalid text to partners. Agent outputs are parsed as JSON, but only the symbol string, action and scalar outcome are stored. Chain-of-thought and raw completions are not retained.

## Context separation

Each agent owns its own conversation history. A helper receives the requester’s symbol string, public scene and its own identity, but never the requester’s private goal or the other helper’s unobserved completion. After simultaneous action selection, the environment sends each agent only its own action, public joint outcome and scalar reward. The owner receives its goal and both visible actions. Histories are never copied between agents.

## Readouts and limits

Primary pilot readouts are valid-message rate, schema/action validity, team success by block, and exact message repeatability within each requester-meaning pair. Each block contains all 18 requester-meaning combinations in a seed-shuffled order. The small pilot is exploratory. It cannot establish language emergence because Qwen has prior language competence, the support is small and repeated, and the task is text-described. A later confirmatory design needs a no-feedback baseline, shuffled-message control, fresh-context transfer, broader held-out meanings and a non-linguistic comparison policy.
