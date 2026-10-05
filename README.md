# Metric self-attention

Does replacing the query-key dot product in self-attention by a negative L^p distance score change what a transformer can learn?
The project studies how the head size D (the query-key dimension per head) interacts with the choice of p. The main experiment is a
small character-level language model on Tiny Shakespeare.

**Status (2026-10-05): the project is being restarted from scratch.** The first version of the code and its pilot are shelved in
`shelved/pilot_v1/`, with notes that explain what was built, what was found, and why it was set aside.

* `docs/project_specification.md`: the current proposal (a draft, edited before any run is launched).
* `docs/project_specification_v1_original_design.md`: the earlier, broader proposal, kept as a record.
* `data/tinyshakespeare.txt`: the corpus (1,115,394 characters, md5 `6fb458f1232090904fb40fe944165e91`).
* `shelved/`: the first version, including the shelved recall task (MQAR). See `shelved/README.md`.
* `requirements.txt`: the pinned environment (torch 2.2.2, numpy below 2).

The design decisions are kept in the project guide with the research notes. No live code exists yet.
