# Emergent redundancy and protocol repair

This study asks whether noisy communication creates pressure for a learned code to acquire error-correcting redundancy, and whether that process differs from simply changing an old code.

The task has four private meanings (two binary factors) and two local resource subtasks. Four workers rotate across episodes while sender identity remains hidden. Messages are delivered simultaneously before the first action, so timing cannot explain a form difference.

- `dual2`: two binary slots, four raw messages, the no-redundancy baseline;
- `triple2`: three binary slots, eight raw messages, an overcomplete coordinate channel in which a learned code can acquire Hamming separation;
- `atomic8`: one eight-valued token, eight raw messages, a capacity-matched control with no internal coordinate structure.

The noise parameter is the independent flip probability for binary slots. For `atomic8`, substitution probability is `1-(1-p)^3`, matching the probability that at least one of three binary coordinates would be corrupted. Each seed first trains a clean parent for every form. Child conditions then compare:

- `scratch`: sender and all workers learn from scratch under the noisy channel;
- `worker_only`: replace worker 0 and freeze the parent sender and incumbent workers;
- `coadapt`: replace worker 0 and update the sender and new worker together.

The key readouts are live−silent, clean versus noisy natural return, incumbent-worker change, sender codebook drift, and minimum pairwise Hamming distance. A triple2 endpoint is labelled an error-correcting candidate only when its noisy natural return is at least 0.60 and its learned codebook has minimum Hamming distance at least 2. This label is a readout, not a prior: no codebook or error-correction rule is supplied to the agents.
