# Prompt history: analysing and writing agent

| | |
|---|---|
| **Role** | Analysed the results of the experiment with the user, ran the statistical tests, wrote the paper in TeX, revised it over several review rounds, tidied the repository and produced the README and avatar. |
| **Model** | Claude (Anthropic). The session was configured for `claude-sonnet-5-5`. |
| **Interface** | Claude Code in a cloud session linked to the user's computer through the Claude desktop app. |
| **Dates** | 2026-10-08 to 2026-10-09 (the earlier part of the session has no recoverable timestamps). |
| **Repository state at the end** | The commit that adds this folder. Earlier commits by this agent: `9eedd90`, `56ec125`, `c8e1972`, `6c63402`, `2984e28`, `bc3af60`. |

## How this file was made

The session ran out of context once and was continued from a summary. The transcript on disk covers only the part after that point.

- **Part A** covers the first part of the session. Its prompts are reconstructed from the compaction summary. Text in quotation marks in the summary is reproduced word for word, except where an entry says it is a fragment or a paraphrase. The times are not recoverable.
- **Part B** covers the second part of the session. Its prompts are copied word for word from the transcript, with timestamps. Images that the agent opened itself are not prompts and are left out.
- **Appendix** reproduces the handover prompt that started the session, recovered from the git history.

Each entry gives the prompt and a one- or two-sentence summary of the response.

## Contents

**Part A: earlier in the session (reconstructed)**
- [A1. Handover prompt for the analysing agent](#a1)
- [A2. Early analysis-stage instructions](#a2)
- [A3. Seed variance and Claim A](#a3)
- [A4. Approve the tests and write the paper](#a4)
- [A5. Spot checking](#a5)
- [A6. First review notes](#a6)
- [A7. Check the new bib entries](#a7)
- [A8. Clean up stale files](#a8)
- [A9. T-test the other way round?](#a9)
- [A10. Second review](#a10)
- [A11. Key-cloud picture](#a11)
- [A12. Do not recapitulate the transformer](#a12)
- [A13. Remove the second motivation](#a13)
- [A14. Superposition as motivation?](#a14)
- [A15. Make the change](#a15)

**Part B: second part of the session (verbatim)**
- [B1. Hold on. Did we ever find out whether superposition is ever ...](#b1)
- [B2. I just don't want to use "superposition" in the wrong sense,...](#b2)
- [B3. Update the Adler reference](#b3)
- [B4. I have some additional questions about the write-up. I will ...](#b4)
- [B5. OK, I guess it's fine. But in Section 2.2 it's probably more...](#b5)
- [B6. Change it to: "chosen so that the scores of a typical query ...](#b6)
- [B7. In Section 2.3. Is the Tiny Shakespeare dataset really autho...](#b7)
- [B8. Ah, OK, so he did put them together. OK, let's leave it as i...](#b8)
- [B9. In Section 2.3 it says "no run diverged". Is this standard t...](#b9)
- [B10. Let's leave it as is.](#b10)
- [B11. You're such a pessimist! Why aren't you happy to just look a...](#b11)
- [B12. OK fine, add a short sentence to the limitations as you sugg...](#b12)
- [B13. The questions are done and I'm happy with the write-up now. ...](#b13)
- [B14. Track the results files](#b14)
- [B15. Let's now organise the repo a bit better. For example, the p...](#b15)
- [B16. Commit (always do this after a significant change from now o...](#b16)
- [B17. Write a snappier, punchier README that is more likely to gra...](#b17)
- [B18. Excellent. Now sweep over the README and other docs and make...](#b18)
- [B19. Now, I want a cute avatar for this project to go in the top-...](#b19)
- [B20. I don't really like any of these. Be a bit bolder and looser...](#b20)
- [B21. Let's refine F. An eye for attention is good. Can we work th...](#b21)
- [B22. Let's iterate on FB1. I just want the ruler to look a bit mo...](#b22)
- [B23. S3 is great, let's use that!](#b23)
- [B24. Now write a ~50 word summary of the project, for my CV. You ...](#b24)
- [B25. I want to add prompt history to the repo. Since there were s...](#b25)

- [Appendix: the handover prompt](#appendix-the-handover-prompt)

---

## Part A: earlier in the session

<a id="a1"></a>
### A1. Handover prompt for the analysing agent

- **Time:** earlier in the session (time not recoverable)
- **Source:** written by the implementing agent

**Prompt**

```text
Not typed by the user. It was written by the previous (implementing) agent as docs/handover_prompt.md and is reproduced in full in the appendix at the end of this file. It defined my role and rules: analyse the results, do not train, do not change the design, commit only if asked, ask when unclear.
```

**Response:** Replied with my understanding of the project, the integrity checks I would run and my questions.

<a id="a2"></a>
### A2. Early analysis-stage instructions

- **Time:** earlier in the session (time not recoverable)
- **Source:** compaction summary (reconstructed)

**Prompt**

```text
[paraphrased in the compaction summary; the original wording is not recoverable] Save a summary of the project in write-up/. Do not begin the full write-up until told. Carry out no statistical tests until they have been discussed. Make integrity checks. Make surface graphs. Compare training times. Organise the discussion around three themes. Discuss the choice of tests.
```

**Response:** Wrote the project summary, ran the integrity checks, drew surface and time graphs and discussed the choice of tests with the user.

<a id="a3"></a>
### A3. Seed variance and Claim A

- **Time:** earlier in the session (time not recoverable)
- **Source:** compaction summary (reconstructed)

**Prompt**

```text
[two fragments, quoted in the compaction summary and truncated there]
"Maybe I'm not understanding something. Don't we want to incorporate the seed variance…"
"I think for Claim A, I would prefer a cell-by-cell comparison…"
```

**Response:** Revised the statistical plan: Claim A is tested cell by cell with a Holm correction within each head size, and the seeds are the paired blocks.

<a id="a4"></a>
### A4. Approve the tests and write the paper

- **Time:** earlier in the session (time not recoverable)
- **Source:** compaction summary (reconstructed)

**Prompt**

```text
[attached the reference paper PDF] I agree and approve all of this. Go ahead and perform the tests. Commit your findings to your working memory / the project documentation. Once you've done this, it's time to write the paper. Go ahead and do this. Here is some guidance: * The paper should be written in TeX and compiled to PDF. * I have attached a reference paper. Use this for determining the tone and general structure, but don't follow it too closely if the needs of this project diverge. * Refer to Claude's general preferences for guidance on technical writing. * Most important: consider heavy use of appendices for technical details, including both detailed results and mathematical background. In this way, the main body of the text should be relatively short and streamlined.
```

**Response:** Ran tests 2 and 3 with the untied and unmasked contrasts, recorded the findings in the project documents and wrote the first version of the paper in TeX with a compiled PDF.

<a id="a5"></a>
### A5. Spot checking

- **Time:** earlier in the session (time not recoverable)
- **Source:** compaction summary (reconstructed)

**Prompt**

```text
Do your full spot checking first, then let me know when it's ready for me to review
```

**Response:** Checked the numbers in the paper against the raw data and reported that it was ready.

<a id="a6"></a>
### A6. First review notes

- **Time:** earlier in the session (time not recoverable)
- **Source:** compaction summary (reconstructed)

**Prompt**

```text
Here are my initial notes. Revise and let me know when it is ready for another review: * Change the title to: "Distance attention at small head sizes for a character-level language model". * The abstract should be more big-picture. Fewer details, more qualitative, more conceptual. * The introduction reads as defensive, trying to overly-justify our choice of topic. State the question first, and move the paragraph about prior work later. * I believe that L^p should actually be \ell^p since we're working with vectors, not functions. * Section 2.1: clarify that c is different for each arm. * Section 2.5: "The seed is the replicate." Such slogan-like sentences are inappropriate for a research paper. Sweep through and edit the entire paper to deal with this. * The formatting of the references is poor. Use the same referencing style as in the example paper I attached. * There are fairly few references. Make sure sufficient references are included e.g. for the dataset and for various bits of background. * The appearance of large numbers (e.g. training errors, p-values, etc.) within paragraphs should be reduced as much as possible: tables are the proper place for these. Sweep through the paper and implement
```

**Response:** Implemented all the notes: new title, qualitative abstract, question-first introduction, \ell^p notation, per-arm c, no slogan sentences, references in the reference paper's style with more entries, numbers moved to tables.

<a id="a7"></a>
### A7. Check the new bib entries

- **Time:** earlier in the session (time not recoverable)
- **Source:** compaction summary (reconstructed)

**Prompt**

```text
Check the new bib entries yourself and correct if necessary.
```

**Response:** Checked every reference against arXiv or publisher pages and fetched the missing author names.

<a id="a8"></a>
### A8. Clean up stale files

- **Time:** earlier in the session (time not recoverable)
- **Source:** compaction summary (reconstructed)

**Prompt**

```text
Clean up stale writeup files (delete if necessary). Actually, delete stale files throughout the reop [sic]
```

**Response:** Deleted the stale write-up files and other stale files.

<a id="a9"></a>
### A9. T-test the other way round?

- **Time:** earlier in the session (time not recoverable)
- **Source:** compaction summary (reconstructed)

**Prompt**

```text
Shouldn't we do the t-test the *other way round*? To be able to say that the tied dot product is better than the lp?

Followed by: Ah, I see. Disregard what I said
```

**Response:** Explained that the paired gap is the same test with the sign reversed, so nothing changed.

<a id="a10"></a>
### A10. Second review

- **Time:** earlier in the session (time not recoverable)
- **Source:** compaction summary (reconstructed)

**Prompt**

```text
I've completed the second review. Implement the following changes: * Introduction: Explain the "boundary of the key cloud" business in more down-to-Earth language. * Introduction: "Compared with the effect of attention itself" - Expand upon what exactly is meant here. * Introduction: There should be a bigger spacing gap underneath the 3 bullet points. * Section 2: Add a bit of a warm-up before launching straight into the mathematics. Tell the reader that we're taking a transformer, but making a change inside the self-attention blocks. * End of Section 2: Add a diagram visualising the training grid (I believe such a diagram already exists somewhere in the repo). * Section 2.1: Change "Two references use" into "Two additional reference arms use". * Section 3.1: The statistical discussion here is too terse. Don't compromise on detail, but take more time and space to explain what is going on, so that it is more comfortable for an average mathematically-literate reader. * Section 3.3: Change "we note this pattern and do not test it" to "we note this pattern without carrying out formal statistical tests" or similar. * Appendix A.1: I don't understand the geometric discussion here. Explain it more carefully, using formulae and figures.
```

**Response:** Implemented all nine changes, including a warm-up in Section 2, a training-grid figure, a fuller statistical explanation and a rewritten geometry appendix with level-set and interior-key figures.

<a id="a11"></a>
### A11. Key-cloud picture

- **Time:** earlier in the session (time not recoverable)
- **Source:** compaction summary (reconstructed)

**Prompt**

```text
For the introduction, add a picture illustrating the key cloud and the boundary, and correspondingly cut down on the current wordy explanation
```

**Response:** Added a key-cloud figure to the introduction and shortened the text.

<a id="a12"></a>
### A12. Do not recapitulate the transformer

- **Time:** earlier in the session (time not recoverable)
- **Source:** compaction summary (reconstructed)

**Prompt**

```text
We should assume that the reader is familiar with the transformer architecture. Sweep over the paper and rewrite any passages which consist of unnecessarily recapitulating the way that architecture works
```

**Response:** Removed the passages that restated how attention works.

<a id="a13"></a>
### A13. Remove the second motivation

- **Time:** earlier in the session (time not recoverable)
- **Source:** compaction summary (reconstructed)

**Prompt**

```text
Remove the second geometric motivation from the introduction
```

**Response:** Removed the D = 1 and D = 2 coincidence from the introduction.

<a id="a14"></a>
### A14. Superposition as motivation?

- **Time:** earlier in the session (time not recoverable)
- **Source:** compaction summary (reconstructed)

**Prompt**

```text
For the geometric motivation, I thought there was a bigger motivation we could give, related to the ML concept of superposition. When the query-key space is small, superposition is hard, so it's difficult to encode relationships, and distances provide a way out. What do you think of this motivation?
```

**Response:** Discussed the idea with two numerical checks (score margins on a circle and a greedy packing count) and qualifications, such as the equal-norm equivalence of squared Euclidean distance and the dot product.

<a id="a15"></a>
### A15. Make the change

- **Time:** earlier in the session (time not recoverable)
- **Source:** compaction summary (reconstructed)

**Prompt**

```text
Excellent. Make these changes to the paper, and really highlight superposition throughout as a motivating idea
```

**Response:** Rewrote the paper around superposition and added Appendix A.2 with the two tables. This framing was later replaced (entry B2).

---

## Part B: second part of the session

<a id="b1"></a>
### B1. Hold on. Did we ever find out whether superposition is ever ...

- **Time:** 2026-10-09 16:21 UTC (17:21 BST)
- **Source:** transcript, verbatim

**Prompt**

```text
Hold on. Did we ever find out whether superposition is ever about query-keys rather than values? Is that what the Elhage / Adler references are about?
```

**Response:** Checked how the paper cites Elhage et al. and Adler, and read Adler's abstract. Answered that Elhage is about features in activations, not query-key space, and that Adler is about the key-query channel but through embedding superposition and the total key dimension. Proposed a more accurate wording and made no change.

<a id="b2"></a>
### B2. I just don't want to use "superposition" in the wrong sense,...

- **Time:** 2026-10-09 16:23 UTC (17:23 BST)
- **Source:** transcript, verbatim

**Prompt**

```text
I just don't want to use "superposition" in the wrong sense, which I think is what we're doing. Perhaps it's safer to just talk about "crowding of the query-key space" and only mention the relationship in passing
```

**Response:** Replaced the "superposition" framing by "crowding of the query-key space" in the abstract, introduction, Section 2, Discussion, Related Work and Appendix A.2. Superposition is now mentioned only in passing. Recompiled, synced to the repo and updated the project guide.

<a id="b3"></a>
### B3. Update the Adler reference

- **Time:** 2026-10-09 16:25 UTC (17:25 BST)
- **Source:** transcript, verbatim

**Prompt**

```text
Update the Adler reference
```

**Response:** Updated the Adler reference to the published ICML 2026 paper (PMLR 306, pages 591 to 626), with the citation key renamed.

<a id="b4"></a>
### B4. I have some additional questions about the write-up. I will ...

- **Time:** 2026-10-09 16:27 UTC (17:27 BST)
- **Source:** transcript, verbatim

**Prompt**

```text
I have some additional questions about the write-up. I will go through these one at a time. Do not make any changes to the write-up without asking: I want direct answers from you first.

First, in Section 2.2 and Appendix C. I thought that in the end the score scale was found empirically, and not through the formula (which I think is even described wrongly in Appendix C, since it should depend on the head size).

Look back at the code and figure out how we found c
```

**Response:** Read msa/calibrate.py and the paper's Section 2.2 and Appendix C. Answered that c is found empirically as the inverse of the pooled standard deviation of the raw scores at initialisation, that the only formula is an after-the-fact sanity check, and that I found no error. No change made.

<a id="b5"></a>
### B5. OK, I guess it's fine. But in Section 2.2 it's probably more...

- **Time:** 2026-10-09 16:29 UTC (17:29 BST)
- **Source:** transcript, verbatim

**Prompt**

```text
OK, I guess it's fine. But in Section 2.2 it's probably more honest to say that c "aims to" make the scores have standard deviation 1, no?
```

**Response:** Agreed that "aims to" is more honest, because the constant fixes only a pooled root mean square at initialisation. Proposed two wordings and asked which to use.

<a id="b6"></a>
### B6. Change it to: "chosen so that the scores of a typical query ...

- **Time:** 2026-10-09 16:30 UTC (17:30 BST)
- **Source:** transcript, verbatim

**Prompt**

```text
Change it to: "chosen so that the scores of a typical query have standard deviation about 1 at initialisation"
```

**Response:** Replaced the sentence in Section 2.2 with the requested wording, recompiled and synced.

<a id="b7"></a>
### B7. In Section 2.3. Is the Tiny Shakespeare dataset really autho...

- **Time:** 2026-10-09 16:31 UTC (17:31 BST)
- **Source:** transcript, verbatim

**Prompt**

```text
In Section 2.3. Is the Tiny Shakespeare dataset really authored by Karpathy? I remember watching a video of his, and I didn't get the sense that the dataset had been put together by him
```

**Response:** Checked the char-rnn README and Karpathy's blog post. Answered that the text is Shakespeare's and that Karpathy distributed it, but that nothing documents who selected the 1.1 million characters. Offered a rewording.

<a id="b8"></a>
### B8. Ah, OK, so he did put them together. OK, let's leave it as i...

- **Time:** 2026-10-09 16:32 UTC (17:32 BST)
- **Source:** transcript, verbatim

**Prompt**

```text
Ah, OK, so he did put them together. OK, let's leave it as it is then
```

**Response:** Left Section 2.3 unchanged, with a note that the source for the small file is the repository.

<a id="b9"></a>
### B9. In Section 2.3 it says "no run diverged". Is this standard t...

- **Time:** 2026-10-09 16:32 UTC (17:32 BST)
- **Source:** transcript, verbatim

**Prompt**

```text
In Section 2.3 it says "no run diverged". Is this standard terminology and is it standard practice to point this out?
```

**Response:** Explained that "diverged" is standard terminology and that reporting it is common practice, and noted that the paper's criterion (loss worse than a uniform guess) is loose. Offered a short gloss. No change made.

<a id="b10"></a>
### B10. Let's leave it as is.

- **Time:** 2026-10-09 16:34 UTC (17:34 BST)
- **Source:** transcript, verbatim

**Prompt**

```text
Let's leave it as is.

Finally, I noticed that each training curve converges (i.e. flattens out), at least when you eyeball it. Is this worth mentioning, or not? What I mean is: it certifies that we iterated for enough steps during training
```

**Response:** Computed the late-training improvement from the logged curves (about 0.008 bpc over the last 500 steps) and explained that the cosine decay forces the curves to flatten. Recommended a sentence in the Limitations and not a claim of convergence.

<a id="b11"></a>
### B11. You're such a pessimist! Why aren't you happy to just look a...

- **Time:** 2026-10-09 16:35 UTC (17:35 BST)
- **Source:** transcript, verbatim

**Prompt**

```text
You're such a pessimist! Why aren't you happy to just look at it and say that it looks like it's converged?
```

**Response:** Conceded that the curves are nearly flat and proposed the wording "an approximately converged budget but not convergence".

<a id="b12"></a>
### B12. OK fine, add a short sentence to the limitations as you sugg...

- **Time:** 2026-10-09 16:35 UTC (17:35 BST)
- **Source:** transcript, verbatim

**Prompt**

```text
OK fine, add a short sentence to the limitations as you suggseted
```

**Response:** Added the sentence to the Scale paragraph of the Limitations, recompiled and synced.

<a id="b13"></a>
### B13. The questions are done and I'm happy with the write-up now. ...

- **Time:** 2026-10-09 16:36 UTC (17:36 BST)
- **Source:** transcript, verbatim

**Prompt**

```text
The questions are done and I'm happy with the write-up now. Commit
```

**Response:** Committed the write-up folder (commit 9eedd90).

<a id="b14"></a>
### B14. Track the results files

- **Time:** 2026-10-09 16:38 UTC (17:38 BST)
- **Source:** transcript, verbatim

**Prompt**

```text
Track the results files
```

**Response:** Committed the results files (commit 56ec125).

<a id="b15"></a>
### B15. Let's now organise the repo a bit better. For example, the p...

- **Time:** 2026-10-09 16:39 UTC (17:39 BST)
- **Source:** transcript, verbatim

**Prompt**

```text
Let's now organise the repo a bit better. For example, the paper is no longer a draft and so shouldn't be in the "draft" folder. Look around and clean things up, get rid of stale files and generally put things in order, ready for public consumption
```

**Response:** Asked two questions (what to do with shelved/ and with the internal working files). Then deleted shelved/ and the stale files, moved the paper to paper/ and the analysis findings to docs/, fixed the script paths, rewrote the README and removed agent-specific rules from the docs.

<a id="b16"></a>
### B16. Commit (always do this after a significant change from now o...

- **Time:** 2026-10-09 16:45 UTC (17:45 BST)
- **Source:** transcript, verbatim

**Prompt**

```text
Commit (always do this after a significant change from now on)
```

**Response:** Committed the tidy-up (commit c8e1972) and recorded the standing rule in the project guide.

<a id="b17"></a>
### B17. Write a snappier, punchier README that is more likely to gra...

- **Time:** 2026-10-09 16:46 UTC (17:46 BST)
- **Source:** transcript, verbatim

**Prompt**

```text
Write a snappier, punchier README that is more likely to grab attention. Include a very prominent link to the full write-up near the top
```

**Response:** Rewrote the README around a headline result, a prominent link to the paper and the gap figure (commit 6c63402).

<a id="b18"></a>
### B18. Excellent. Now sweep over the README and other docs and make...

- **Time:** 2026-10-09 16:49 UTC (17:49 BST)
- **Source:** transcript, verbatim

**Prompt**

```text
Excellent. Now sweep over the README and other docs and make sure they are up-to-date and consistent with the final project design
```

**Response:** Updated the docs and README to match the final project: recorded the outcome, stated which analyses the paper uses, replaced agent working rules with a neutral record, switched to the paper's notation and regenerated the design figures (commit 2984e28).

<a id="b19"></a>
### B19. Now, I want a cute avatar for this project to go in the top-...

- **Time:** 2026-10-09 16:52 UTC (17:52 BST)
- **Source:** transcript, verbatim

**Prompt**

```text
Now, I want a cute avatar for this project to go in the top-right of the README. Prototype a few examples and show me them for approval
```

**Response:** Drew four avatar concepts (A to D) and showed them on a contact sheet.

<a id="b20"></a>
### B20. I don't really like any of these. Be a bit bolder and looser...

- **Time:** 2026-10-09 16:53 UTC (17:53 BST)
- **Source:** transcript, verbatim

**Prompt**

```text
I don't really like any of these. Be a bit bolder and looser, anything that captures the idea of metric attention. We don't need to see the diamond shapes of l1 intervals.
```

**Response:** Drew four bolder and looser concepts (E to H) built on measuring metaphors.

<a id="b21"></a>
### B21. Let's refine F. An eye for attention is good. Can we work th...

- **Time:** 2026-10-09 16:55 UTC (17:55 BST)
- **Source:** transcript, verbatim

**Prompt**

```text
Let's refine F. An eye for attention is good. Can we work the metric in more clearly? Perhaps a ruler or something like that?
```

**Response:** Developed concept F into three ruler variants and then two polished candidates (FB1 and FB2).

<a id="b22"></a>
### B22. Let's iterate on FB1. I just want the ruler to look a bit mo...

- **Time:** 2026-10-09 16:57 UTC (17:57 BST)
- **Source:** transcript, verbatim

**Prompt**

```text
Let's iterate on FB1. I just want the ruler to look a bit more like a ruler. Perhaps it should be straighter, or have little centimetre markings along it, or something. Try a few things out
```

**Response:** Drew four ruler styles (S1 to S4) with centimetre marks and numerals, centred them and tested them at 72 pixels.

<a id="b23"></a>
### B23. S3 is great, let's use that!

- **Time:** 2026-10-09 16:58 UTC (17:58 BST)
- **Source:** transcript, verbatim

**Prompt**

```text
S3 is great, let's use that!

Save it to the repo in an appropriate place (get rid of the other prototypes). Place it in the top-right of the README at a decent size, and commit
```

**Response:** Exported S3 as assets/avatar.svg and assets/avatar.png, placed it at the top right of the README and committed (commit bc3af60). Deleted the other prototypes.

<a id="b24"></a>
### B24. Now write a ~50 word summary of the project, for my CV. You ...

- **Time:** 2026-10-09 17:01 UTC (18:01 BST)
- **Source:** transcript, verbatim

**Prompt**

```text
Now write a ~50 word summary of the project, for my CV. You should outline the project itself, its motivation, its findings, and the technical skills required to achieve it. Paste the result here
```

**Response:** Wrote a 51-word CV summary and pasted it into the reply.

<a id="b25"></a>
### B25. I want to add prompt history to the repo. Since there were s...

- **Time:** 2026-10-09 17:10 UTC (18:10 BST)
- **Source:** transcript, verbatim

**Prompt**

```text
I want to add prompt history to the repo. Since there were several agents involved, here is how we're going to do this:

1) You will create a folder in the repo to contain the prompt history.

2) I will ask each agent to create a markdown file with their prompt history, in the created folder. You will create a prompt for this purpose. You will also add your own prompt history to the folder.

3) Once all the agents have done this, I will let you know and you will organise the prompt history into one, easily navigable file.
```

**Response:** This request. Created the prompt_history/ folder with a README, a prompt for the other agents and this file.

---

## Appendix: the handover prompt

This is the prompt that started the session. It was written by the implementing agent and kept in the repository as `docs/handover_prompt.md` until the tidy-up commit `c8e1972`. It is reproduced from commit `56ec125`.

```text
# Handover prompt for the analysing agent

You are taking over the Metric Self-Attention project at the analysis stage. The design is settled, the code is written and tested, and the full run has finished. Your task is to analyse the results with the user and to discuss them. You do not change the design and you do not run any training.

## The project

The project tests whether replacing the query-key dot product in self-attention by a negative L^p distance helps, and how the answer depends on the head size D. The exponents are p in {1, 1.5, 2, 3, 4, 8, infinity}. The setting is a small character-level language model on Tiny Shakespeare. The grid has 47 cells: nine arms at D in {2, 4, 8, 12, 16}, plus a uniform-attention control and an unmasked untied dot-product reference at D = 16. Each cell has five seeds. The outcome is the final validation bits per character.

## Reading order

Read these in the repository root (the folder metric-self-attention) before replying:

1. README.md
2. docs/project_outline.md
3. docs/design_details.md
4. docs/implementation_plan.md (section 8.1 for the test-1 form, section 13 for the coding decisions, section 14 for the record of the run)
5. docs/results_guide.md (file locations and schemas, the analysis definitions, points to keep in mind, write-up requirements, open items)
6. The project memory guide areas/metric-self-attention.md, then index.md

If a memory note and a repository document disagree, the document is current.

## Rules

- Ignore the shelved/ folder. Ignore the memory file areas/metric-self-attention-code.md.
- Do not train, scan or pilot. Do not edit anything under results/ except results/analysis/. You may rerun `python execute_project.py analyze` freely.
- The design, including the statistical plan, is fixed. If you think a change is needed, say so and ask the user.
- Commit locally only if the user asks. Never push. Never use `git add -A`.
- Write in formal prose with short, self-contained clauses. Add no extra tests, caveats or options unless they change a conclusion.
- Talk options through carefully and in detail. Ask the user when anything is unclear.
- Record user decisions in the memory guide as [stated] lines only after the user confirms them.

## Facts to keep in mind

- No run diverged. The rate scan chose 3e-3 in 46 of 47 cells and 1e-2 for the uniform control, so per-arm rate tuning is weak.
- The user has decided that no finer learning-rate scan is made. All conclusions rest on the existing run.
- The final weights are committed. The rest of results/ is untracked.
- The median seed sd is about 0.0096 bpc.
- Test 6 (equivalence) is not coded. Its margin is deferred and is needed only if an equivalence claim is made.
- The write-up must make the statements listed in design_details section 10, including that the baseline is non-standard because of the diagonal mask and that rates were tuned per arm on the reporting set.

## Your first reply

Do not analyse yet. Reply with:

1. A short summary of your understanding of the project, the design and the run.
2. The integrity checks you will make on results/ (for example that all 235 main runs are present, that each result.json agrees with its log.csv, and that a final.pt reloads).
3. Your questions for the user, including what they want from the discussion.

## If you cannot reach the user's files

If the device bridge is unavailable, ask the user to attach these files: everything in results/analysis/tables/, results/scan_table.csv, results/selected_rates.json and results/calibration/calibration_summary.csv.
```
