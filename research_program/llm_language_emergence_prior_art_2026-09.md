# LLM language emergence: prior art and research gap

**Reviewed:** 2026-09-20. This is a focused novelty update for the Qwen agent study, not yet a systematic review of all work on human language origins or computational language evolution.

## Closest primary sources

| Work | Setup and finding relevant here | Consequence for our project |
|---|---|---|
| Stengel-Eskin et al., “GlossoGen: Emergent Language in Complex Multi-Agent LLM Interactions,” arXiv:2609.01491v1 (submitted 2026-09-01) | Two LLM roles solve a cooperative, partially observed, sequential Veyru treatment task. The authors vary a 150 vs. 2000 character budget and access to an unconstrained postmortem discussion. They report non-English, productive conventions under pressure plus postmortem for several proprietary models; Qwen3-32B and Llama-3.3-70B do not independently develop new languages in their tested setup, but can learn existing conventions from interaction history. They also study replacement/transfer to newcomers. | “Complex multi-agent task + pressure + language change,” postmortem convention discussion, compositional morphology, and cultural transmission are already claimed. These should not be our primary novelty claims. Their main scenario has two distinct roles; our narrower candidate gap is whether an opaque convention allocates content and actor-selection information among interchangeable helpers. |
| Anonymous, “Emergence of Machine Language in LLM-Based Agent Communication,” under review at ICLR 2026 (OpenReview PDF `zy06mHNoO2`) | A two-agent referential game with a predefined alphabet and target-vs-distractor choice. Its agent design retrieves semantically similar words for the speaker and uses structural proximity for listener decoding. It reports shared mappings for 541 objects after four rounds, with up to three attempts per communication, and claims compositionality, generalization, morphemes, and polysemy. | “LLMs can invent opaque strings that communicate” and “machine language is compositional” are direct prior claims. The engineered retrieval and structural-decoding mechanism differs from our pretrained contexts learning through task outcomes, but the paper’s status and method must be checked again before submission. |
| Kouwenhoven, Peeperkorn & Verhoef, “Searching for Structure: Investigating Emergent Communication with Large Language Models,” COLING 2025 | LLMs play a classical referential game. Initially holistic codes gain structure through interaction; generational transmission can improve learnability while producing degenerate vocabularies. | Referential games, structural change, and transmission already have direct LLM precedents. A paper must show what role-dependent ecological tasks add. |
| Lian, Verhoef & Bisazza, “NeLLCom-X: A Comprehensive Neural-Agent Framework to Simulate Language Learning and Group Communication,” CoNLL 2024 | Extends a neural-agent language-learning framework with role alternation, group communication, and group-size effects. | Group size and interaction topology are not novel by themselves. The potential distinction is applying causal information-allocation tests to pretrained LLM contexts and measuring who/what is encoded in a shared convention. |
| Boldt & Mortensen, “Searching for the Most Human-like Emergent Language,” EMNLP 2025 | Uses optimization over signaling-game hyperparameters and human-likeness/transfer criteria. | Hyperparameter search for human-like emergent languages is also prior art. A single sweep over message budget or task difficulty would be weak novelty. |

## What currently looks overclaimed or under-specified

An opaque symbol repertoire and successful coordination do not by themselves establish language. At minimum, claims about language structure need evidence that form–meaning mappings are shared across senders and receivers, survive partner replacement, and generalize to meanings not seen as complete combinations. Perplexity under an English model can measure distance from English but cannot establish grammar. Likewise, sample mutual information can be high under sender-specific codes or message collapse; it needs controls and held-out decoding.

LLMs share a large natural-language prior even when each has an isolated context. They also receive researcher-written task instructions. For this reason, our current use of “emergence” should be operationalized as a convention acquired through interaction, not as human-like de novo language origin. Later experiments should compare ordinary LLM contexts with learners that have less shared prior or controlled artificial semantic descriptions.

## Plausible differentiation for the current program

The potentially useful causal question is narrower than “can LLMs form a language?” In a three-agent cooperative task with two interchangeable workers, vary whether each worker is told that it is the selected actor and vary how precisely the environment scores what happened. Measure whether shared symbols encode the task payload (object, property, destination), actor allocation, or both. This tests how division of labor and feedback shape the information carried by a convention. The v3 all-wait result motivates this diagnostic decomposition.

This is a candidate gap, not yet a novelty guarantee. NeLLCom-X already studies group interaction and size in neural agents; GlossoGen already studies complex LLM teamwork and transmission; the ICLR submission already studies opaque LLM languages and compositionality. Before an ICLR submission, the project will need a broader search, a precise comparison table, held-out compositional transfer, newcomer or partner-swap tests, and replication across model families. An open-weight model or a three-agent count alone is not a sufficient contribution.

## Primary links

- [GlossoGen, arXiv paper](https://arxiv.org/abs/2609.01491) · [author-linked code](https://github.com/agencyenterprise/GlossoGen)
- [Emergence of Machine Language in LLM-Based Agent Communication, OpenReview submission](https://openreview.net/pdf?id=zy06mHNoO2)
- [Searching for Structure, ACL Anthology](https://aclanthology.org/2025.coling-main.667/)
- [NeLLCom-X, ACL Anthology](https://aclanthology.org/2024.conll-1.19/)
- [Searching for the Most Human-like Emergent Language, ACL Anthology](https://aclanthology.org/2025.emnlp-main.1188/)
