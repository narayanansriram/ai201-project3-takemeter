# TakeMeter — planning.md

> Complete this document before writing any classification code.
> Your taxonomy, decision rules, and collection rules become the blueprint for annotation, prompt design, and the fine-tuning pipeline.
> This document will be reviewed as part of your submission.

---

## Community

**Chosen subreddit:** r/TrueFilm

**Why r/TrueFilm:**
r/TrueFilm is a strong fit because the community produces all three label types naturally and in roughly comparable volume. New-release threads produce reactions; monthly discussions produce analysis; debate threads ("Is X actually a masterpiece?") produce hot takes. The variance in post quality is high, which is exactly what makes the classification task meaningful. r/criterion was considered but rejected because its community self-selects for analytical cinephiles, making the dataset harder to balance — hot_take and reaction examples are scarcer there, and a large fraction of posts are about the physical collection rather than the films.

---

## Label Taxonomy

Three labels cover ≥90% of opinion-expressing comments from film-specific discussion threads. No "other" bucket is used — ambiguous cases are resolved by the decision rules below.

---

### `analysis`

**Definition:** Makes a structured argument using specific evidence: named scenes, cinematographic techniques, directorial choices, named comparator films, primary source quotes, or the film's internal logic. The post could stand alone as criticism. The claim is falsifiable — a reader could go watch the film and test whether the evidence holds.

**Key signal:** Remove the film references. If the argument collapses, it's analysis. If the comment still makes the same claim without them, it's something else.

**Examples:**
- Citing Snyder's own words ("to hone in on things he likes to look at") as primary source evidence for a technical claim about Army of the Dead's cinematography
- Using the Stalker internal logic (the room grants truest desires, not stated ones) to argue Porcupine's suicide makes sense within the film's rules
- Comparing Nolan's visual language across named films (Following, Memento, Prestige) to identify a specific shift after Dark Knight Rises

---

### `hot_take`

**Definition:** States a confident opinion without meaningful evidence. Bold, often provocative, but asserts rather than argues. Claims exist about the film independent of the viewer's experience, but nothing checkable is offered to support them.

**Key signal:** The opinion is about the film (not the viewer), but the argument could not be contested on the film's own terms because no specific evidence is cited.

**Examples:**
- "Nolan only really has one genre" — categorical claim with no film evidence
- "Everything else is overrated" — bold assertion, no support
- Structured-sounding argument (complexity over clarity → emotional disconnect) that names the pattern but cites no specific film or scene

---

### `reaction`

**Definition:** An immediate emotional or experiential response to a specific viewing experience. Little to no argument. Stays in the first person and describes what the viewer felt rather than making claims about what the film is.

**Key signal:** Every sentence describes the viewer's experience ("I felt," "I was confused," "it destroyed me"). The opinion is about the viewer's experience of the film, not about the film itself.

**Examples:**
- "Just finished watching this movie. I need to talk about it because it really blew my mind."
- "I really really didn't like or care about any of the characters" — stays first-person throughout
- "I was so let down by this movie" — disappointment narrative

---

## Critical Edge Cases & Decision Rules

These rules were developed by working through borderline examples before annotating the full dataset.

---

### Rule 1: The film-reference removal test
The primary test for analysis vs. hot_take. Ask: if you removed all the film references, would the argument collapse? If yes → analysis. If the comment still makes the same claim without them → hot_take. The Army of the Dead comment cites Snyder's own words and a VFX breakdown as load-bearing evidence; removing them destroys the argument. The "Nolan only has one genre" comment loses nothing when its topic list is removed.

---

### Rule 2: Quoted dialogue does not automatically make something analysis
Quoting specific dialogue is often decorative, not argumentative. The test is whether the quote is being *interpreted* (what does this line do to the scene? how does it undercut the character arc?) or merely *pointed at* ("this line cringe so hard"). The Interstellar docking scene comment quotes two lines but uses them only to name the feeling — no mechanism is proposed for why the lines fail. → hot_take.

---

### Rule 3: Structured argumentation without specific evidence is still a hot_take
A logically structured framework ("he tends to overvalue complexity over clarity, which causes X, so audiences feel Y") is still a hot_take if no specific film, scene, or technique is cited to ground the claim. The presence of a logical skeleton does not make something analysis — it makes it an articulate hot_take.

---

### Rule 4: reaction vs. hot_take — first person vs. third person
The boundary turns on whether the opinion is about the film or about the viewer's experience.
- "This film isn't profound" → hot_take (claim about the film, exists independently of any viewer)
- "I didn't find the film profound" → reaction (claim about the viewer's experience)
- "I really really didn't like or care about any of the characters... I just don't find the allegory all that profound" → reaction (stays first-person throughout; every evaluative sentence is filtered through "I")

---

### Rule 5: Hot_take framing does not change a comment's label
A comment that opens with "I'm sure I'll get shit on for this" or "I know this is a hot take" and then proceeds to make a specific, evidence-based argument is still analysis. The attitude is a wrapper, not the content. Similarly, a comment that opens analytically but whose body is all assertion is a hot_take. Always label based on what the body of the comment is doing.

---

### Rule 6: Short comments and context-dependence
Comments under ~30 words are flagged as potential low-quality training examples, not because they are unlabelable, but because they carry too little signal for a model to learn from. "I agree with the exception of his animated films. His style seems to fit the medium well." is a hot_take by definition, but at ~20 words it's near-useless as a training example. Flag these in the notes column.

---

### Rule 7: Analysis can look like a rant
Messy prose, strong tone, and chaotic structure don't disqualify a comment from being analysis. The Army of the Dead comment is stream-of-consciousness but every claim is attached to something verifiable (Snyder's own words, the VFX breakdown, named comparator films and their cinematographers). The question is always whether the evidence is doing argumentative work.

---

### Rule 8: Film technique named ≠ analysis
"Every shot means something. There's all kinds of camera tricks and special effects to help tell the story visually." Names technique without demonstrating it — no specific shots, no named techniques, no scenes. This is reaction (enthusiasm about the film's craft), not analysis.

---

## Collection Rules

These rules were set before data collection began and applied consistently throughout.

1. **Comments only, not top-level posts.** Top-level posts have too many recommendation requests, news shares, and essay-length takes that require different annotation logic. Comments in film-specific discussion threads are almost always opinion-expressing and fit the three labels cleanly.

2. **Film-specific threads only.** Only collect from threads where the prompt is about a specific film — "What did you think of X?", "Official discussion: [film]", "Is [film] actually a masterpiece?", "I just watched [film]". Skip any thread where the prompt is "who are your favorite directors?" or "what should I watch?" — these generate watchlist/recommendation content that doesn't fit the taxonomy.

3. **Minimum length: ~30 words.** Comments below this threshold are too thin to carry useful signal. They are usually agreements ("I agree"), reactions to the parent comment, or fragments. These are labeled `no_fit` and excluded from training data.

4. **No scraping.** Comments were collected by copy-paste from specific threads. Source thread URL is tracked in the notes column for provenance.

5. **no_fit is not a label — it's an exclusion.** Comments that are recommendation lists, viewing advice, questions without opinions, or off-topic content are marked `no_fit` and excluded from the training set. They are not trained on.

---

## Hyperparameter Decision

**Decision:** Train for 5 epochs instead of 3 if validation accuracy is still climbing at epoch 3.

**Reasoning:** With ~140 training examples after the 70/15/15 split, the model may underfit with only 3 epochs. If validation accuracy has not plateaued by epoch 3, run 2 additional epochs and record the accuracy at each step. If accuracy plateaus before epoch 3, stop early.

**What to document:** Validation accuracy at each epoch checkpoint, final decision, and whether the extra epochs helped.

---

## Evaluation Metrics

**Primary metric: per-class F1** (harmonic mean of precision and recall per label).

Overall accuracy is insufficient here because the dataset has mild class imbalance (analysis is ~46%) — a model that always predicts analysis would score ~46% accuracy without learning anything. F1 penalizes both false positives and false negatives per class, so a model that ignores hot_take entirely will show F1 ≈ 0 for that class even if overall accuracy looks acceptable. Per-class F1 forces the model to demonstrate it has learned all three boundaries, not just the majority class.

Confusion matrix is reported alongside F1 to show which label pairs are being confused — overall F1 doesn't reveal whether errors are concentrated at one boundary (e.g., analysis ↔ hot_take) or distributed evenly.

---

## Success Criteria

- Per-class F1 ≥ 0.65 on all three labels
- Fine-tuned model accuracy beats zero-shot baseline by at least 10 percentage points
- If fine-tuning barely moves the needle: inspect the confusion matrix for analysis ↔ hot_take confusion first — this is the most likely source of annotation noise

---

## Baseline Prompt Design

The zero-shot Groq prompt must include:

1. All three label definitions verbatim (from this document)
2. One canonical example per label (show the text + label, not just the label name)
3. The decision rule that the model is most likely to need: the film-reference removal test (Rule 1) and the first-person vs. third-person test (Rule 4)
4. Strict output instruction: "Reply with only one word: analysis, hot_take, or reaction."

If parsing fails on more than 10% of responses, add: "Reply with only one word. Do not include punctuation, explanation, or any other text."

---

## Data Split Plan

| Split | Size | Notes |
|-------|------|-------|
| Train | ~140 | 70% of usable examples |
| Val   | ~30  | 15% — used for epoch decisions |
| Test  | ~30  | 15% — held out until final evaluation |

Current dataset: 96 usable examples (45 analysis, 28 hot_take, 23 reaction). Need ~104 more before the split is reliable. Target distribution on final dataset: no label above 70%.

---

## Annotation Log

| Batch | Examples collected | Thread types | Notes |
|-------|-------------------|--------------|-------|
| 1 | 25 usable (from 25 total) | Nolan visual language, Eyes Wide Shut, Barry Lyndon, The Handmaiden | 73% analysis — imbalanced |
| 2 | 55 usable (from 55 total) | Same thread types | 73% analysis — still imbalanced |
| 3 | 96 usable (from 102 total, 6 no_fit) | Added: IJW threads, "critically acclaimed films I don't get", Marty Supreme, Wuthering Heights 2026, Central Station | 47% analysis, 29% hot_take, 24% reaction — healthy |

---

## AI Tool Plan

**Annotation assistance (Milestone 1):**
Used Claude to pre-label batches of 20–25 comments given the three definitions and one example per label. Reviewed every pre-label independently — disagreements were used to sharpen decision rules (Rules 1–8 above). All borderline cases were resolved manually and documented in the notes column of comments_labeled_v2.xlsx.

**Zero-shot baseline (Milestone 3):**
Will give Claude/Groq the prompt design spec (definitions + examples + output format) and ask it to generate the classifier function. Will verify output parsing works before running the full evaluation loop.

**Fine-tuning (Milestone 4):**
Will use the dataset from data/comments.csv. Will document the hyperparameter decision (epoch count) with accuracy curves.
