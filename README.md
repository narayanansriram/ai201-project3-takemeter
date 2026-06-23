# TakeMeter

**Demo:** [Loom walkthrough](https://www.loom.com/share/d231ab583f3648f89d92100414a8b309)

A fine-tuned text classifier for r/TrueFilm Reddit comments. Given a comment, the system assigns one of three labels: `analysis`, `hot_take`, or `reaction`. The notebook fine-tunes DistilBERT on annotated examples and compares it to a Groq zero-shot baseline.

**Stretch goals completed:** Error Pattern Analysis · Confidence Calibration

---

## Setup

1. Open `ai201_project3_takemeter_starter_clean.ipynb` in Google Colab.
2. Set runtime to **T4 GPU** (Runtime → Change runtime type → T4 GPU).
3. Add your Groq API key to Colab Secrets (🔑 icon, left sidebar) as `GROQ_API_KEY`.
4. Upload `data/comments_balanced.csv` when Section 1 prompts for a file.
5. Run all cells top to bottom.

---

## Label Taxonomy

| Label | Definition |
|-------|-----------|
| `analysis` | Structured argument with specific evidence — named scenes, techniques, named comparator films, or primary sources. Falsifiable: remove the film references and the argument collapses. |
| `hot_take` | Confident opinion about the film with no meaningful evidence. Bold and assertive but unsubstantiated. |
| `reaction` | First-person emotional or experiential response. Reports what the viewer felt, not claims about what the film is. |

**Examples per label:**

`analysis`:
- *"Harrison Ford from 1980–1985. In a row, he did: Empire Strikes Back, Raiders of the Lost Ark, Blade Runner, Return of the Jedi, Temple of Doom, and Witness. That's an insane run, ending with a nomination for Best Actor. All of those, except the one he was nominated for, are classics."*
- *"The Departed is, in a number of ways, a better-made movie than the original Infernal Affairs, but remaking an existential drama as a crowd-pleasing action movie means it just replaces the deeper emotions of Infernal Affairs with more violence and swearing and hallmark Scorsese tricks."*

`hot_take`:
- *"Nolan only really has one genre. His recent films are technically impressive but visually interchangeable — the look of Dunkirk could swap with Interstellar and nothing would feel off."*
- *"I think they work better as short films or small pieces that you take in and experience. But as a feature length movie it just feels a bit much. I appreciate the style, the craft, editing, sound, all very well implemented."*

`reaction`:
- *"I saw Jaws back when it was originally released and the shark effect was freaking terrifying. Back then there was no instant access to real shark footage, so for many of us this was the first time we saw a shark presented that way — and it really worked on me."*
- *"I absolutely love the constant plot twists in The Handmaiden and I had the time of my life watching it in cinemas. I remember being marvelled by it for days. I also loved it had a more 'optimistic' ending."*

Full taxonomy, 8 decision rules, and edge cases: [planning.md](planning.md).

---

## Dataset

**Source:** r/TrueFilm — IJW (I Just Watched) threads, director discussion threads, "critically acclaimed films I don't get" threads. Top-level posts and recommendation threads excluded. Minimum ~30 words per comment.

**Total labeled:** 200 usable comments. 6 additional `no_fit` comments (off-topic, too short, not about film content) excluded before training.

**Label distribution (raw):**

| Label | Count | % |
|-------|-------|---|
| `analysis` | 92 | 46% |
| `hot_take` | 51 | 25.5% |
| `reaction` | 57 | 28.5% |

**Balanced training file (`comments_balanced.csv`):** `hot_take` and `reaction` oversampled via random duplication to match `analysis` at 92 each — 276 rows total. Used for all training runs after the 1st.

**Split:** 70% train / 15% val / 15% test (stratified by label) — handled automatically by the notebook.

**Annotation:** Pre-labeled in batches using Claude with the three definitions and one canonical example per label. Every pre-label reviewed independently. All 8 decision rules in [planning.md](planning.md) emerged from borderline cases where the pre-label and my label differed — those disagreements were the most useful annotation data.

**Three genuinely difficult examples:**

1. *"Visually stunning and psychologically haunting movie. Got the pleasure of seeing it on the big screen at the Berkeley Art Museum..."* — Reads like a reaction (personal viewing experience, first-person) but contains film-specific craft language ("visually stunning", "psychologically haunting") that could pass for analysis. Labeled `analysis` based on specific craft claims. Model predicted `reaction` (confidence 0.38) — lowest-confidence wrong prediction in the test set.

2. *"Most movies written by Kaufman (with a few exceptions like Being John Malkovich). They are all exceptionally well made and I agree that the man is a genius, BUT I can always feel that there is something..."* — Personal taste expressed through a career overview. The "I can always feel" framing is reaction-coded but the claim ("something missing across his filmography") is a hot_take about the work, not a feeling about a viewing. Labeled `reaction`; model predicted `hot_take` (confidence 0.43).

3. *"He's always seemed like a director who's focused more on the technical aspects of film. That's not a bad thing. I'd put James Cameron, George Lucas, and Robert Zemeckis in that category too."* — Names three specific directors as comparators, which looks like evidence. But no argument is made — the names are illustrative, not load-bearing. Labeled `hot_take`; model predicted `analysis` (confidence 0.67).

---

## Fine-Tuning Approach

**Base model:** `distilbert-base-uncased` (HuggingFace)

**Training setup:** Google Colab T4 GPU. Default notebook pipeline: tokenization → stratified 70/15/15 split → Trainer API.

**Hyperparameter decisions across runs:**

| Run | Dataset | Epochs | Learning rate | Fine-tuned accuracy |
|-----|---------|--------|---------------|-------------------|
| 1 | `comments.csv` (imbalanced) | 3 | 2e-5 | 46.7% |
| 2 | `comments_balanced.csv` | 3 | 2e-5 | 64.3% |
| 3 | `comments_balanced.csv` | 5 | 2e-5 | 66.7% |
| 4 | `comments_balanced.csv` | 5 | 3e-5 | 71.4% |
| 5 | `comments_balanced.csv` | 6 | 3e-5 | 73.8% |

The single biggest gain came from oversampling minority classes (run 1 → 2, +17.6pp). Run 1 was a degenerate model — it predicted `analysis` for every example because that was the majority class at 46%. Balancing the training data forced the model to learn all three boundaries.

---

## Baseline

**Model:** `llama-3.3-70b-versatile` via Groq API (zero-shot, no fine-tuning).

**Prompt:** Definitions for all three labels with one canonical example each, instructing the model to output only the label name.

```
You are classifying comments from r/TrueFilm, a Reddit community for serious film discussion.
Assign each comment to exactly one of the following categories.

analysis: Makes a structured argument using specific evidence — named scenes, cinematographic
techniques, directorial choices, named comparator films, or primary source quotes. The argument
is falsifiable: remove the film references and the claim collapses.
Example: "Harrison Ford from 1980-1985. In a row, he did: Empire Strikes Back, Raiders of the
Lost Ark, Blade Runner, Return of the Jedi, Temple of Doom, and Witness..."

hot_take: States a confident opinion about a film without meaningful evidence. Bold and assertive
but unsubstantiated — claims exist about the film independent of the viewer's experience, but
nothing checkable is offered to support them.
Example: "Nolan only really has one genre. His recent films are technically impressive but
visually interchangeable..."

reaction: An immediate emotional or experiential response to a specific viewing. Stays in the
first person throughout — describes what the viewer felt, not claims about what the film is.
Example: "I saw Jaws back when it was originally released and the shark effect was freaking
terrifying..."

Respond with ONLY the label name. Do not explain your reasoning.
```

---

## Evaluation Results

**Overall accuracy:**

| Model | Accuracy |
|-------|----------|
| Zero-shot baseline (Groq llama-3.3-70b-versatile) | 78.6% |
| Fine-tuned DistilBERT (epochs=6, lr=3e-5) | **73.8%** |

**Per-class metrics — fine-tuned model (final run):**

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| `analysis` | 0.69 | 0.79 | 0.73 | 14 |
| `hot_take` | 0.71 | 0.71 | 0.71 | 14 |
| `reaction` | 0.83 | 0.71 | 0.77 | 14 |
| **macro avg** | **0.75** | **0.74** | **0.74** | 42 |

**Confusion matrix (final run, test set = 42 examples):**

|  | Predicted: analysis | Predicted: hot_take | Predicted: reaction |
|--|---------------------|---------------------|---------------------|
| **True: analysis** | 11 | 2 | 1 |
| **True: hot_take** | 3 | 10 | 1 |
| **True: reaction** | 2 | 2 | 10 |

**Error analysis — 3 misclassified examples:**

**1. True: `reaction` → Predicted: `analysis` (confidence: 0.82)**

> *"The shark effect I assure you in its time was freaking terrifying. (Yes, I saw the film way, waaaaay back in the Stone Age when it was originally released... cough-old-fart-cough). Back then, unlike now..."*

The model's highest-confidence wrong prediction. This is a first-person reaction comment — the commenter is describing their personal experience seeing Jaws on original release. But the comment includes historical contextual framing ("back then, unlike now") and references the specific production context of the shark effect, which pattern-matches to the kind of contextual evidence that appears in `analysis` examples. The model learned that historical/contextual knowledge signals analysis, but didn't learn that first-person framing overrides that signal.

**2. True: `hot_take` → Predicted: `analysis` (confidence: 0.67)**

> *"He's always seemed like a director who's focused more on the technical aspects of film. That's not a bad thing. I'd put James Cameron, George Lucas, and Robert Zemeckis in that category too. Decent storyteller..."*

Naming three specific directors (Cameron, Lucas, Zemeckis) as comparators looks structurally like evidence — and the model treats it as such. But the names are purely illustrative; no argument is made about any of them. The model hasn't learned the distinction between names-as-decoration and names-as-load-bearing-evidence. This is the hardest boundary in the taxonomy to operationalize from surface features alone.

**3. True: `reaction` → Predicted: `hot_take` (confidence: 0.63)**

> *"Been on a Michael Haneke run. Cache (Hidden) this week - the anxiety builds from scene one and never really lets up. Then Funny Games, which is a completely different kind of uncomfortable. Both leave..."*

The comment uses declarative third-person framing ("the anxiety builds", "a completely different kind of uncomfortable") even though it's describing a personal viewing experience. The model correctly identifies the absence of first-person language and calls it `hot_take`. The label is `reaction` because the experiential context (a personal Haneke viewing run, week-by-week) makes clear this is personal response — but the model has no access to that framing and relies on surface pronouns and phrasing.

**Most common confusion:** `analysis` ↔ `reaction` (6 errors across both directions). Comments with historical or contextual framing get called `analysis` even when first-person; long analytical-feeling comments occasionally get called `reaction`.

**Sample classifications (fine-tuned model, final run):**

| Text (truncated) | True label | Predicted | Confidence | Correct? |
|------------------|-----------|-----------|------------|----------|
| "I'm sure there was a lot of kneejerk pearl clutching criticism of the nudity, but it's also very explicitly a film about the male gaze..." | `analysis` | `analysis` | 0.76 | ✓ |
| "I saw that one in the theater back when it came out. I don't think I'd seen the original all the way through as a teenager..." | `reaction` | `reaction` | 0.80 | ✓ |
| "Been on a Michael Haneke run. Cache (Hidden) this week - the anxiety builds from scene one and never really lets up. Then Funny Games..." | `reaction` | `hot_take` | 0.63 | ✗ |
| "Sicario (2015) Dir. Denis Villeneuve Gonna keep this short but sweet. Villeneuve is one of my favorite directors in the game right now..." | `reaction` | `reaction` | 0.65 | ✓ |
| "yeh i too cherish mandy. finally something w interesting visuals. the music is great. story actually has a heart..." | `hot_take` | `hot_take` | 0.80 | ✓ |

The first correct prediction (analysis, 0.76) is reasonable: the comment frames nudity in terms of the film's thematic intent ("explicitly a film about the male gaze"), which is a falsifiable structural claim about what the film is doing — exactly what `analysis` captures.

---

## Confidence Calibration

Predictions bucketed by confidence score on the 42-example test set:

| Confidence range | Predictions | Accuracy |
|------------------|-------------|----------|
| 40%–60% | 11 | 64% |
| 60%–75% | 13 | 62% |
| 75%–100% | 17 | **94%** |

The model's confidence scores are meaningful. When confidence is above 75%, the model is correct 94% of the time — nearly all errors occur in the low-to-mid confidence range (40–75%). This means the confidence score is a reliable signal: a high-confidence prediction can be trusted, while a prediction below 75% should be treated skeptically. The 62% accuracy in the 60–75% bucket is close to chance for a 3-class problem (33% random baseline), confirming that mid-confidence predictions represent genuine uncertainty at label boundaries.

---

## Reflection

**What the model learned vs. what I intended:**

I intended the model to learn the *argumentative structure* of each label — whether a claim is backed by falsifiable evidence, stated without support, or grounded in personal experience. What it actually learned was closer to surface-level signals: historical/contextual language → `analysis`, declarative third-person claims → `hot_take`, first-person pronouns → `reaction`.

This works most of the time because those signals correlate with the intended distinctions. But it breaks down at the boundary cases: a reaction comment written in declarative style gets called `hot_take`; a hot_take that names specific films or directors gets called `analysis`. The model learned the correlates of my labels, not the labels themselves — which is exactly what you'd expect from 200 examples of a subtle, subjective task.

The Groq baseline outperforming the fine-tuned model (78.6% vs. ~71%) is consistent with this. The baseline has access to natural language definitions and can reason about argumentative structure from the prompt. The fine-tuned model has to infer that structure from patterns in 200 examples — and 200 is not enough to learn something this subtle reliably.

---

## Spec Reflection

**One way planning.md helped:**

Writing the 8 decision rules before annotating forced me to confront the hardest boundary early: the difference between naming a film as evidence versus naming it as decoration. Rule 1 (film-reference removal test) was the most useful annotation tool — I applied it on nearly every borderline case. Having it written down meant I applied it consistently across 200 examples rather than drifting toward intuition.

**One way implementation diverged from what I expected:**

I expected the fine-tuned model to outperform the zero-shot baseline — that's the standard outcome of fine-tuning. It didn't, and it didn't come close in the first run (46.7% vs. 73.3%). The failure was a class imbalance problem I hadn't anticipated: with `analysis` at 46% of the dataset, the model learned to predict it constantly rather than learn the other two boundaries. Rebalancing via oversampling (run 2) fixed the degenerate behavior but the baseline advantage held throughout all four runs. The spec didn't prepare me for the possibility that a general-purpose LLM with a well-written prompt would be harder to beat than expected on a 200-example dataset.

---

## AI Usage

**Instance 1 — Annotation assistance:**
Used Claude to pre-label batches of 20–25 comments using the three definitions and one canonical example per label. Reviewed every pre-label independently. All 8 decision rules in [planning.md](planning.md) emerged from borderline cases where Claude's label and my label initially differed — those disagreements were the most useful annotation data produced in the entire project.

**Instance 2 — Classifier prompt construction:**
Used Claude to draft the initial SYSTEM_PROMPT for the Groq baseline (Section 5 of the notebook). The draft matched the definitions in planning.md but I revised the examples — the first draft used hypothetical examples, and I replaced them with actual comments from the dataset that clearly illustrated each label. The final prompt is in the Baseline section above.

**Instance 3 — Error pattern analysis:**
After run 4, pasted all 11 wrong predictions into Claude and asked for common themes. It identified the analysis/reaction boundary as the primary failure mode and noted the historical-context signal as a likely cause. I verified this by reading each wrong prediction individually — the pattern held, but two of the eleven didn't fit the pattern (the Kaufman comment and the Eternal Sunshine comment, which were more about first-person vs. third-person framing).
