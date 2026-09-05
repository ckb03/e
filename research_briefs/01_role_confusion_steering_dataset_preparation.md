# Role-Confusion Steering — Dataset Preparation

## Scope

This first implementation is intentionally narrow.

Goal:

> Test whether activation steering can reduce prompt-injection success when the injected text appears inside a **Tool** result but internally shifts toward another role-like representation.

For now:

- use the same 5 roles as the role-confusion paper/demo:
  - `system`
  - `user`
  - `cot`
  - `assistant`
  - `tool`
- do **not** add `developer`
- only evaluate injections whose **true architectural role is Tool**
- the runtime defense knows that the span is Tool output
- the runtime defense does **not** know:
  - whether the Tool result is injected
  - where the injection begins
  - which role/style it is imitating

Keep four kinds of data separate:

1. `D_repr`: learn role representations
2. `D_clean`: calibrate normal Tool behavior
3. `D_attack_dev`: select layer and steering hyperparameters
4. `D_test`: final frozen evaluation only

---

# 1. `D_repr` — role-representation dataset

## Purpose

Used to compute, at every model layer:

- 5-way linear role probe
- role centroids
- pairwise steering vectors
- SVD / continuous role subspace

This is **not** attack data.

## Recommended size

Start with:

- 150 base texts total
- 75 from C4 validation
- 75 from Dolma3

Each base text is rendered under all 5 roles:

\[
150 \times 5 = 750
\]

model forward passes.

This should be enough for an initial experiment and is close to the upstream demo scale. Increase to 250–400 base texts only if:

- probe accuracy is unstable
- pairwise vectors vary strongly across random seeds
- SVD spectrum / role geometry is noisy

## Source

Match the upstream role-confusion setup as closely as practical:

- 50% C4 validation
- 50% Dolma3
- deterministic shuffle / fixed random seed
- truncate each base sequence to 512 content tokens
- use actual GPT-OSS/native role formatting

Example roles:

```text
system
user
cot
assistant
tool
```

## Construction

For each base text `x_i`, create:

```text
render_as_system(x_i)
render_as_user(x_i)
render_as_cot(x_i)
render_as_assistant(x_i)
render_as_tool(x_i)
```

The content must be **identical** across the five copies.

Only the role wrapper/tag should change.

## Important constraints

- exclude role delimiter/tag tokens when collecting content activations
- keep content-token alignment across role variants
- do not include instructional/attack-specific text deliberately
- split by **base text**, never by individual token or role copy
- avoid allowing one long text to dominate vector estimates

## Suggested split

For 150 base texts:

```text
105 base texts -> representation train
20 base texts  -> representation dev
25 base texts  -> representation test
```

All five role copies of one base text must remain in the same split.

Use:

- `repr_train`: fit probe and vectors/subspace
- `repr_dev`: probe calibration / regularization choices if needed
- `repr_test`: verify probe generalization and role geometry

---

# 2. `D_clean` — normal Tool-result calibration set

## Purpose

A normal Tool result may naturally look partly User-like, Assistant-like, etc.

Therefore do not assume:

\[
p(\text{Tool}|h)=1
\]

for every clean Tool token.

`D_clean` estimates the normal distribution of Tool activations and is used to avoid steering benign Tool output unnecessarily.

## Recommended size

Start with:

\[
30\text{ clean Wikipedia pages}
\]

Optionally increase to 40 if thresholds look unstable.

Suggested split:

```text
20 pages -> calibration
10 pages -> clean sanity/validation
```

## Source and construction

Use exactly the same Wikipedia distribution and page-fetching logic as the existing agent eval:

```python
load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train",
    streaming=True,
)
```

Apply the same:

- shuffle logic
- HTML fetch
- max-page-size constraint
- HTML cleanup
- agent task/tool environment

But:

- use pages NOT present in the frozen final test manifest
- insert NO injection
- save the raw clean HTML
- freeze the selected page list before inspecting results

## What to record

Run the normal agent/tool setup.

For all Tool-result content tokens, save:

```text
page_id
token_position
layer
activation
true_role = tool
```

Later this is used to estimate:

### Soft-pairwise calibration

For each layer:

\[
p_\ell(r|h), \quad r\in
\{\text{system,user,cot,assistant,tool}\}
\]

on clean Tool tokens.

### Continuous calibration

For each layer:

- Tool role-space mean
- Tool role-space covariance
- distribution of distance from the normal Tool region

---

# 3. `D_attack_dev` — attack development set

## Purpose

Used for:

- selecting the steering layer
- selecting steering strength `alpha`
- selecting continuous-subspace dimension `k`
- selecting any gating/smoothing hyperparameters

This must be separate from the final frozen test set.

## Recommended size

Because the full agent eval is expensive, start small.

Use:

\[
30\text{ NEW Wikipedia pages}
\]

Each page gets the two existing attack families:

```text
base-injection
cot-forgery-injection
```

Therefore:

\[
30 \times 2 = 60\text{ attack cases}
\]

Also keep the clean counterpart for every page.

Recommended split:

```text
10 pages × 2 = 20 attacks -> layer selection
10 pages × 2 = 20 attacks -> hyperparameter tuning
10 pages × 2 = 20 attacks -> development validation
```

So:

### `D_layer`

```text
10 clean pages
20 attacked cases
```

### `D_tune`

```text
10 clean pages
20 attacked cases
```

### `D_devval`

```text
10 clean pages
20 attacked cases
```

If compute is especially constrained, start with 8 pages for layer selection instead of 10.

---

# 4. How to generate `D_attack_dev`

Reuse the current manifest-generation structure.

The existing pipeline:

1. samples Wikipedia pages
2. downloads the HTML
3. checks size / valid `</body>`
4. chooses an injection prompt
5. inserts it immediately before `</body>`
6. freezes all resulting HTML into JSONL

This is suitable for development data.

## Requirements

Use:

- a new seed
- pages disjoint from final `D_test`
- same page-size limits
- same HTML insertion point
- same agent/eval environment
- same injection families

For this initial steering feasibility experiment it is acceptable to reuse the same injection-template pool as the final eval.

This means the final test primarily measures generalization to unseen webpages/content, not unseen attack-template families.

Later, if steering works, construct a stronger unseen-template test.

## Prefer balanced assignment

Instead of pure:

```python
injection = rng.choice(prompts)
```

prefer balancing template usage across development cases if easy.

However, do not delay the experiment for perfect balancing.

The primary first question is:

> Does activation correction reduce attack success at all?

---

# 5. `D_test` — final frozen evaluation

Use the existing frozen 200-case manifest unchanged.

It contains:

```text
100 Wikipedia pages × 2 variants = 200 attack cases
```

with:

```text
base-injection
cot-forgery-injection
```

Do NOT use these cases for:

- probe fitting
- vector/subspace extraction
- layer selection
- alpha tuning
- k selection
- gating threshold selection
- choosing between variants of the algorithm

Only run the final frozen configuration here.

---

# 6. Dataset overlap rules

Enforce at least:

```text
D_repr content != D_clean / attack webpages
D_clean URLs != D_attack_dev URLs
D_attack_dev URLs != D_test URLs
```

The most important rule is:

```text
D_attack_dev URLs ∩ D_test URLs = empty
```

Also keep every clean/attacked version of the same webpage in the same development split.

Do not put:

```text
clean(page_X)
```

in layer selection while:

```text
attacked(page_X)
```

appears in tuning or final test.

---

# 7. Suggested manifests

Create explicit frozen files:

```text
eval_data/
    steering_repr_manifest.jsonl
    steering_clean_manifest.jsonl
    steering_attack_layer_manifest.jsonl
    steering_attack_tune_manifest.jsonl
    steering_attack_devval_manifest.jsonl
```

Do not regenerate them automatically every run.

Each attack-development row should contain at least:

```json
{
  "case_id": 0,
  "split": "layer",
  "variant": "base-injection",
  "source_url": "...",
  "url": "...",
  "injection_id": "...",
  "injection_sha256": "...",
  "html": "..."
}
```

Keep a corresponding clean page identifier so attacked and clean activations can be paired when useful.

---

# 8. Quality checks before running expensive experiments

Before freezing development manifests, assert:

## Representation data

- all 5 role variants contain identical content tokens
- role wrappers are model-native
- tag tokens are identifiable/excludable
- no base text occurs in multiple splits

## Clean data

- no injection is present
- pages do not overlap final test
- agent can successfully process the page
- Tool result reaches the model without truncating most page content

## Attack-dev data

- injection is present in final Tool-result context
- injection is not truncated away
- clean counterpart exists
- URL does not overlap final test
- both attack variants are represented
- no automatic filtering based on whether the undefended model succeeds

Do not discard attacks simply because baseline ASR is low.

---

# 9. Minimum first-run dataset budget

Use this if the goal is rapid feasibility testing:

```text
D_repr:
    150 base texts
    750 role-wrapped forward passes

D_clean:
    30 Wikipedia pages

D_attack_dev:
    30 Wikipedia pages
    60 attack cases total

    20 attacks -> layer
    20 attacks -> tune
    20 attacks -> dev validation

D_test:
    existing 200 attacks
    run only after freezing the method
```

Most `D_repr` work is cheap model prefill / activation extraction and does not require the full agent eval.

The expensive agent evaluations are therefore limited to small 20-case development blocks until the final test.
