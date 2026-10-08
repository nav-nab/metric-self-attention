# Claims removed from the live project

On 2026-10-08 the user separated the live project completely from version 1. The statements below were carried into the live documents from version 1. They have been removed from the live documents. They are kept here as a record, and none of them is a result of the live project.

## Section 5 of the outline (results from the pilot)

These come from version 1 of the project, which is shelved in `shelved/pilot_v1/`. Each finding is listed there, with its caveats and its relevance to this project, in `shelved/pilot_v1/PILOT_FINDINGS.md`. The full report is `shelved/pilot_v1/docs/pilot_report.md`. Every pilot run used head size 16, and no pilot number describes a small head.

- At head dimension 16, the language-model arms are within seed noise of each other. Tied dot and L2 are indistinguishable, and p = infinity is slightly worse. This is why the project now turns to small heads.
- Background (recall task, now shelved): on the recall task, small head dimension is where the dot product breaks down while the distance arms do not. At larger D every arm solves the task. The distance arms are all near perfect, so the recall task cannot rank the values of p.
- At an adequate learning rate, convergence speed among the L^p arms shows no clear ordering. There is a weak hint that mid values of p learn faster early and that p = 1 and p = infinity learn slower. It rests on one seed.
- Learning rate matters. The dot arm is more sensitive to it at small D than the distance arms.
- The seed-to-seed spread of validation bpc is about 0.01, and the smallest difference detectable with five seeds is about 0.02 (finding 7 in the pilot findings).
- The language model underfits: training loss is well above what a model of this kind could reach. So these experiments probe a regime limited by capacity as much as by attention.

## Lines removed from the limitations section of the outline

- The model is small and underfit, so an effect of the score may be hidden by other limits.
- Seed noise is about 0.01 bpc, so only differences above roughly 0.02 are detectable with five seeds. This figure comes from an earlier version at 2,500 steps and D = 16, and it will be re-measured.

## Other statements removed

- Outline, learning rate: "In the earlier work, the dot arm needed a far higher rate at small D, and a shared low rate gave the false impression that distance was better." The same statement was removed from section 8 of the design details.
- Design details, section 8: "One seed has noise of about 0.01 bpc."
- Design details, section 9: "The noise level of about 0.01 bpc comes from the earlier version, at a different length ... a difference smaller than roughly 0.02 bpc cannot be detected."
- Design details, section 11: the reasons given for shelving the recall task (the distance arms sit at its accuracy ceiling, the dot-product result depends on the learning rate, and further pilots would not end).
