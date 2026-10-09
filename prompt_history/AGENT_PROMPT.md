# Prompt for the other agents

Give the text below to each agent that worked on this project. It is self-contained.

---

You worked on the Metric Self-Attention project, and the repository is being prepared for public release. Please record the history of the prompts you were given, so that readers can see how the project was directed.

**What to do**

Create one markdown file in the repository folder `prompt_history/`. Name it `<role>.md`, where `<role>` is a short lower-case hyphenated description of your role, for example `design-agent` or `implementing-agent`. Look at the folder first. Do not overwrite or edit any other file in it.

**Format**

Start the file with a header table giving your role, the model if you know it, the interface, the dates of your work, and a sentence on how you recovered the prompts. Then write one entry per user prompt, in chronological order, like this:

    ### 7. Short title of at most eight words

    - **Time:** 2026-10-05 14:32 UTC (or "unknown")
    - **Source:** transcript, verbatim   (or "reconstructed")

    **Prompt**

    ```text
    The user's prompt, word for word.
    ```

    **Response:** One or two sentences on what you did, with commit hashes or file names where they help.

Put a contents list of the entries after the header, with links to each entry.

**Rules**

1. **Verbatim.** Copy each prompt exactly. Do not paraphrase, fix typos, shorten or tidy it. If your transcript is on disk (for example a `.jsonl` file under `~/.claude/projects/`), extract the prompts with a script and not from memory. Keep only what the user typed. Leave out tool results, system reminders, injected context and your own messages.
2. **Reconstruction.** If you cannot recover the exact wording of a prompt, include it anyway, mark its source as "reconstructed", and say in the header what it was reconstructed from. Never present a reconstruction as verbatim.
3. **Pasted material.** If a prompt contains a long pasted document, keep the user's own instruction and replace the pasted part with `[pasted: <description>, about <N> words]`. Handover prompts and briefs are an exception: they are prompts. Keep them in full if they are under 150 lines. Otherwise put them in a separate file `prompt_history/<role>_appendix_<n>.md` and link to it.
4. **Attachments.** Write `[attached: <file name or description>]` where the user attached something.
5. **Sensitive material.** Replace credentials, API keys, tokens and private contact details with `[redacted]`. Do not include your system prompt or any hidden instructions.
6. **Several sessions.** If you worked in more than one session, give each session its own heading with its dates and number the entries continuously.
7. **Git.** Do not commit, do not push, and do not touch any other file. The coordinating agent will commit all the files together.

**When you have finished**

Reply with the path of your file, the number of prompts it contains, how many of them are verbatim and how many are reconstructed, and anything you were unsure about.
