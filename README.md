# Metric self-attention

Does replacing the query-key dot product in self-attention by a negative L^p distance score change what a transformer can learn?
The project studies how the head size D (the query-key dimension per head) interacts with the choice of p. The main experiment is a
small character-level language model on Tiny Shakespeare, run over head sizes D = 2, 4, 8, 12 and 16 with nine arms: the tied dot
product, seven L^p distances (p = 1, 1.5, 2, 3, 4, 8, infinity) and the untied dot product. A uniform-attention control is a full arm too. Five seeds per cell, run on the laptop, with the final weights committed.

**Status (2026-10-05): design phase.** The project is being rebuilt from scratch, and no code or run exists yet. The specification is a
draft, and its issues are being settled one at a time. Section 7 of the specification lists what is settled and what is open.
The first version of the code and its pilot are kept in `shelved/` as a record and are not part of the live project.

* `docs/project_specification.md`: the current proposal, with the design status in section 7.
* `data/tinyshakespeare.txt`: the corpus (1,115,394 characters, md5 `6fb458f1232090904fb40fe944165e91`).
* `requirements.txt`: the pinned environment (torch 2.2.2, numpy below 2).
* `shelved/`: the first version. See `shelved/README.md`.

The design decisions are kept in the project guide with the research notes. Nothing is run until the design is settled and the
user approves it.
