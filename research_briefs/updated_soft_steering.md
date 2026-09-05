# Role-Confusion Steering v2 — Probe Training and Improved Soft-Pairwise Algorithm

## Goal

Implement a stronger, cleaner test of the hypothesis:

> Prompt injection inside a true **Tool** result can shift internal activations toward other role-like representations, and correcting that shift can reduce attack success.

This document specifies:

1. how to train a high-quality role probe;
2. how to compute role-to-Tool directions;
3. how to calibrate clean Tool behavior;
4. an improved **soft-pairwise v2** intervention that fixes the main problems in the current implementation;
5. why a probe is still worth training even if the later experiment uses the continuous role-region method.

The runtime defense knows only that the current span is a true Tool result. It does **not** know whether an injection exists, where it begins, or which role it imitates.

---

# 1. Activation site

For the next experiment, use the **same GPT-OSS activation site used by the role-confusion paper/repository** (`all_pre_mlp_hidden_states`, i.e. the pre-MLP hidden state at the chosen layer).

Use the **same activation site for all three things**:

- probe training;
- role-direction extraction;
- runtime steering.

Do not train the probe at one representation and apply the steering direction at another.

The previous post-block location should be treated as an alternative to compare later, not the default, because the current layer-6 probe there was weak.

---

# 2. Representation dataset `D_repr`

## 2.1 Recommended size

Use the paper-scale setup as the minimum serious run:

```text
250 neutral base texts
maximum 1024 content tokens per base text
approximately 25% C4 validation
approximately 75% Dolma3
5 roles
```

A concrete deterministic allocation is:

```text
62 C4 validation texts
188 Dolma3 texts
250 total
```

If this works and representation extraction remains cheap, increase the final representation run to **400–500 base texts**. Do not increase data before first verifying that the paper-scale setup produces a strong probe; poor accuracy at 250 bases is more likely to indicate an extraction/site/formatting problem than simply insufficient data.

## 2.2 Roles

Use exactly:

```text
system
user
cot
assistant
tool
```

Do not add `developer` in this first experiment.

For every base text `x_i`, render the **same content** under all five native GPT-OSS role wrappers:

```text
render_as_system(x_i)
render_as_user(x_i)
render_as_cot(x_i)
render_as_assistant(x_i)
render_as_tool(x_i)
```

Only the role wrapper may change.

## 2.3 Token handling

- truncate source content to at most 1024 content tokens;
- exclude all role/header/delimiter tokens from probe/vector examples;
- preserve alignment of content tokens across the five renderings;
- use all valid aligned content tokens if memory permits;
- do **not** use the previous maximum of only 64 sampled tokens per base;
- if memory becomes limiting, use at least 256–512 deterministic evenly spaced aligned content tokens per base-role rendering.

To prevent long examples from dominating, use base-balanced sample weights. If base `i` contributes `T_i` aligned tokens for each role, give each token weight proportional to:

\[
\frac{1}{T_i}.
\]

The total contribution of each underlying base text should therefore be approximately equal.

## 2.4 Split

Split by **underlying base text before role rendering**:

```text
175 bases -> repr_train
35 bases  -> repr_dev
40 bases  -> repr_test
```

All five role copies of one base text stay in the same split.

Never split individual tokens or role-rendered copies independently.

---

# 3. Train the role probe

At each candidate layer `l`, collect the pre-MLP activation:

\[
h_{i,j,r}^{(l)} \in \mathbb{R}^{d}
\]

for base text `i`, aligned content-token position `j`, and true role `r`.

Train a five-way multinomial linear probe:

\[
z^{(l)}(h)=W_l h+b_l
\]

\[
p_l(r\mid h)=\operatorname{softmax}(z^{(l)}(h))_r.
\]

Use the upstream-style classifier first:

```text
multinomial L2 logistic regression
C = 5e-3
no hidden layers
no nonlinear features
```

Prefer the same cuML/scikit-compatible logistic-regression formulation used upstream rather than an AdamW-trained PyTorch classifier, so optimizer convergence is not another variable.

Do not temperature-scale the probe for the main v2 intervention. The revised algorithm uses **logit margins**, not calibrated probabilities.

## 3.1 Evaluation

Evaluate only on `repr_test`, with base-balanced weighting.

Record:

```text
overall 5-way accuracy
per-role accuracy
confusion matrix
cross entropy
```

Also report results on `repr_dev` during development, but do not use `repr_test` to tune the classifier.

Important sanity check:

- if the paper-aligned site/setup still gives low accuracy, stop before expensive agent evaluation and debug role rendering, token alignment, activation extraction, and hook location.

A probe with approximately 50% accuracy is not good enough to act as the router for soft-pairwise steering.

---

# 4. Compute matched role-to-Tool directions

Use only `repr_train`.

For each base `i` and role `r`, first average aligned content-token activations within that base:

\[
\mu_{i,r}^{(l)}
=
\frac{1}{T_i}
\sum_{j=1}^{T_i}
h_{i,j,r}^{(l)}.
\]

Then compute the matched difference for each wrong role:

\[
d_{i,r\rightarrow T}^{(l)}
=
\mu_{i,T}^{(l)}
-
\mu_{i,r}^{(l)}.
\]

Average equally across bases:

\[
v_{r\rightarrow T}^{(l)}
=
\frac{1}{N}
\sum_i
d_{i,r\rightarrow T}^{(l)}.
\]

Compute:

```text
system -> tool
user -> tool
cot -> tool
assistant -> tool
```

Do not use attack examples to construct these directions.

---

# 5. Recommended role-subspace cleanup of the pair directions

The raw pair vectors may contain off-role noise. Build a low-dimensional role subspace from the same matched representation data.

For each base-role mean:

\[
\bar{\mu}_i
=
\frac{1}{5}\sum_r \mu_{i,r}
\]

\[
x_{i,r}
=
\mu_{i,r}-\bar{\mu}_i.
\]

Stack all `x_{i,r}` and compute SVD. Let `B` contain the top role directions.

For five role centroids, start with:

\[
k=4.
\]

`B` must be orthonormal:

\[
B^\top B=I.
\]

Project each pair vector into this role subspace:

\[
\tilde v_{r\rightarrow T}
=
BB^\top v_{r\rightarrow T}.
\]

Then normalize it:

\[
u_{r\rightarrow T}
=
\frac{\tilde v_{r\rightarrow T}}
{\|\tilde v_{r\rightarrow T}\|+\epsilon}.
\]

The runtime algorithm uses these **unit directions** `u`, not the raw vectors `v`.

This separates:

- which direction to move;
- how large the intervention should be.

It prevents roles with naturally larger raw pair-vector norms from automatically receiving much stronger interventions.

---

# 6. Clean Tool calibration dataset `D_clean`

Use real benign Tool outputs from the same distribution as the agent evaluation.

Recommended:

```text
30 clean Wikipedia pages
20 pages -> calibration
10 pages -> held-out clean sanity check
```

No injection may appear in these pages.

At the exact same activation site and layer used by the probe/steering, collect all true Tool-content activations.

For every clean Tool token `j`, compute the probe logits:

\[
z_{j,r}=W h_j+b.
\]

For each wrong role:

\[
r\in\{system,user,cot,assistant\},
\]

compute the **wrong-role-vs-Tool logit margin**:

\[
m_{j,r}=z_{j,r}-z_{j,T}.
\]

This is preferable to using raw softmax probability because it directly measures evidence for role `r` relative to the known true role Tool.

---

# 7. Standardize wrong-role evidence

For each wrong role `r`, estimate a robust clean Tool baseline from the calibration pages:

\[
c_r=\operatorname{median}(m_{j,r})
\]

\[
s_r=1.4826\cdot
\operatorname{median}
\left(
|m_{j,r}-c_r|
\right)+\epsilon.
\]

Then define token-level standardized abnormality:

\[
q_{j,r}
=
\frac{m_{j,r}-c_r}{s_r}.
\]

Interpretation:

```text
q ~= 0       normal Tool behavior for this role margin
q > 0        more role-r-like than a typical clean Tool token
large q      unusually role-r-like for true Tool content
```

This replaces the old quantity:

\[
p_r-\tau_r.
\]

---

# 8. Smooth role evidence over neighboring tokens

The old algorithm gated every token independently, creating isolated spikes and many zeros inside a coherent injection.

Instead smooth the standardized role evidence over a local window.

Default first implementation:

```text
window size = 32 content tokens
```

For each token:

\[
\bar q_{j,r}
=
\frac{1}{|N_j|}
\sum_{k\in N_j}
q_{k,r},
\]

where `N_j` is the available 32-token local neighborhood around token `j`.

A Gaussian or triangular kernel is also acceptable, but use the same smoothing rule for clean calibration and runtime.

The window is used only for **detection/routing**. The actual correction is still applied separately to each token activation.

---

# 9. Joint clean gate instead of four independent thresholds

Define the local joint confusion score:

\[
S_j
=
\max_{r\neq T}
\bar q_{j,r}.
\]

On the 20 clean calibration pages, run the complete smoothing pipeline and collect all `S_j`.

Set **one joint threshold**:

\[
\beta
=
Q_{0.99}
\left(
S_j\mid clean,\ true=Tool
\right).
\]

Default target:

```text
~1% clean Tool token/window firing rate
```

Optionally test 99.5% if benign interventions are still too frequent.

This is deliberately different from four independent 95th-percentile gates.

Runtime:

\[
S_j\le\beta
\quad\Rightarrow\quad
\Delta h_j=0.
\]

Therefore the new soft-pairwise algorithm still does **not** intervene on every Tool token.

---

# 10. Soft-pairwise v2 routing

For a token whose local region passes the joint gate, compute a soft source-role mixture from the smoothed standardized margins:

\[
\pi_{j,r}
=
\frac{
\exp(\bar q_{j,r}/T_{route})
}{
\sum_{s\neq T}
\exp(\bar q_{j,s}/T_{route})
}.
\]

Default:

\[
T_{route}=1.
\]

This avoids the old behavior where a role contribution becomes exactly zero merely because its probability lies slightly below an arbitrary per-role percentile.

All four wrong roles receive a soft relative weight, but the role with the strongest abnormal evidence dominates.

---

# 11. Build a normalized corrective direction

Combine the normalized pair directions:

\[
d_j^{raw}
=
\sum_{r\neq T}
\pi_{j,r}u_{r\rightarrow T}.
\]

Normalize the mixture:

\[
d_j
=
\frac{
d_j^{raw}
}{
\|d_j^{raw}\|+\epsilon
}.
\]

If:

\[
\|d_j^{raw}\| < 0.1,
\]

treat the routing direction as unstable and set:

\[
\Delta h_j=0.
\]

This avoids amplifying near-cancellation between incompatible role directions.

---

# 12. Bounded intervention magnitude

Do not let raw pair-vector norm determine steering strength.

Define soft anomaly severity:

\[
g_j
=
\begin{cases}
0,&S_j\le\beta\\
1-\exp(-(S_j-\beta)/\gamma),&S_j>\beta.
\end{cases}
\]

Default:

\[
\gamma=1.
\]

Choose an interpretable maximum relative intervention size:

\[
\rho_{\max}.
\]

Then:

\[
M_j
=
\rho_{\max}\|h_j\|g_j.
\]

Finally:

\[
\boxed{
\Delta h_j=M_jd_j
}
\]

and:

\[
\boxed{
h'_j=h_j+\Delta h_j.
}
\]

By construction:

\[
\frac{\|\Delta h_j\|}{\|h_j\|}
\le
\rho_{\max}.
\]

Initial development sweep:

```text
rho_max ∈ {0.0025, 0.005, 0.01}
```

Do not perform a large grid.

This replaces the old unbounded:

\[
\alpha\sum_r w_rv_{r\rightarrow T}.
\]

---

# 13. Complete runtime pseudocode

```python
# Inputs fixed for selected layer:
# W, b
# clean_margin_center[r]
# clean_margin_scale[r]
# beta
# T_route
# gamma
# rho_max
# unit_pair_direction[r] = u[r]
#
# h_tool: [num_tool_tokens, d]
# Only true Tool-content tokens are passed here.

logits = h_tool @ W.T + b                       # [T, 5]

for r in WRONG_ROLES:
    margin[:, r] = logits[:, r] - logits[:, TOOL]
    q[:, r] = (
        margin[:, r] - clean_margin_center[r]
    ) / clean_margin_scale[r]

q_smooth = local_smooth(q, window=32)          # same rule as calibration
S = q_smooth[:, WRONG_ROLES].max(dim=-1)

for j in range(num_tool_tokens):
    if S[j] <= beta:
        continue

    route_logits = q_smooth[j, WRONG_ROLES] / T_route
    pi = softmax(route_logits)

    d_raw = 0
    for k, r in enumerate(WRONG_ROLES):
        d_raw += pi[k] * unit_pair_direction[r]

    d_raw_norm = norm(d_raw)

    if d_raw_norm < 0.1:
        continue

    direction = d_raw / d_raw_norm

    severity = 1.0 - exp(-(S[j] - beta) / gamma)

    max_norm = rho_max * norm(h_tool[j])
    delta = max_norm * severity * direction

    h_tool[j] += delta
```

---

# 14. What to tune and what not to tune

Use `D_attack_dev`, never final `D_test`.

Keep tuning small.

Recommended first choices:

```text
layer:
    chosen from a small set of layers with strong repr_dev probe quality
    then evaluated causally on D_attack_dev

window:
    16 vs 32 tokens

joint clean threshold:
    99% vs 99.5%

rho_max:
    0.0025, 0.005, 0.01

T_route:
    fix at 1 initially

gamma:
    fix at 1 initially
```

Do not simultaneously grid all combinations.

Suggested order:

1. reproduce strong probe;
2. choose 2–3 candidate layers;
3. test window 32, threshold 99%, `rho_max=0.005`;
4. inspect detection localization and intervention norms;
5. only then make a tiny tuning sweep.

---

# 15. Diagnostics required before agent evaluation

For attacked and clean Tool content, log per token:

```text
token position
token text
probe logits
wrong-role-vs-Tool margins
standardized q by role
smoothed q by role
joint score S
joint threshold beta
route weights pi
chosen direction norm before normalization
severity g
||h||
||delta||
||delta|| / ||h||
```

For each page, summarize separately:

```text
injection span
near-injection context
page_other
```

Check:

1. injection regions should have substantially higher joint anomaly score than ordinary page content;
2. isolated HTML/CSS spikes should be suppressed by smoothing;
3. clean Tool intervention rate should approximately match the configured joint false-positive rate;
4. intervention norm can never exceed `rho_max * ||h||`;
5. after steering, Tool probe evidence should generally rise and the dominant wrong-role margin should generally fall.

If these checks fail, do not run the expensive agent sweep.

---

# 16. Why train a probe if using the continuous algorithm?

The **continuous role-region method does not require the probe at runtime**.

Its runtime quantities are instead:

```text
role subspace B
clean Tool role-space mean
clean Tool covariance / whitening transform
joint Tool-region distance threshold
```

So, strictly speaking:

\[
\boxed{
\text{continuous runtime steering can work without }W,b.
}
\]

The reason to train the paper-style probe first is **validation**, not because the continuous formula needs it.

It provides three valuable sanity checks:

### A. Verify that the representation extraction is correct

The paper's main mechanistic result is that role is strongly linearly represented at its chosen activation site.

If a paper-aligned probe trained on paper-aligned data performs poorly, then something is probably wrong with:

```text
role rendering
token alignment
activation site
hook implementation
or data construction
```

Running a continuous role-subspace intervention before resolving that would make a negative result hard to interpret.

### B. Verify that the chosen layer/site actually contains clean role information

The continuous method assumes there is a meaningful role geometry.

A strong role probe is an inexpensive check that such geometry is present.

### C. Use the probe as a mechanistic readout

Even when the probe is not part of runtime steering, measure before/after:

\[
p(Tool\mid h)
\]

and wrong-role logits.

Then test whether the continuous intervention actually moves representations in the intended role sense.

Therefore the recommended workflow for a **continuous-only experiment** is:

```text
1. reproduce/train strong paper-style probe
2. use it to validate representation extraction/site
3. build continuous role subspace
4. run continuous intervention without using probe probabilities as runtime weights
5. retain probe only for diagnostics/readout
```

You do **not** need to first run a full soft-pairwise agent evaluation merely because the probe was trained.

---

# 17. Continuous alternative in one equation

For reference, the continuous method is conceptually separate from soft-pairwise v2.

Project a Tool token into an orthonormal role subspace:

\[
z=B^\top(h-c).
\]

Using clean Tool data estimate:

\[
m_T,\Sigma_T.
\]

Compute whitened Tool distance:

\[
D^2
=
(z-m_T)^\top
(\Sigma_T+\lambda I)^{-1}
(z-m_T).
\]

After optional local smoothing, use a **single joint clean threshold**.

If the state is normal:

\[
D^2\le\tau
\Rightarrow
\Delta h=0.
\]

If it lies outside the normal Tool region, move only the abnormal role-space component back toward the boundary of the clean Tool region, with a relative norm cap.

This method does not ask which discrete wrong role caused the displacement.

---

# 18. Recommended immediate implementation order

```text
1. Switch representation extraction to the paper's pre-MLP site.
2. Build 250-base, 1024-token, 25/75 C4-Dolma D_repr.
3. Train upstream-style multinomial L2 logistic probes.
4. Verify strong repr_test accuracy and confusion matrices.
5. Compute matched role->Tool vectors.
6. Build k=4 role subspace and project/normalize pair directions.
7. Recompute D_clean at the exact same site.
8. Replace probability-percentile gates with standardized logit margins.
9. Add 32-token smoothing.
10. Calibrate one joint 99% clean threshold.
11. Replace raw pair-vector scaling with normalized direction mixture.
12. Bound ||delta||/||h|| by rho_max.
13. Run token-level localization diagnostics.
14. Only if diagnostics look sensible, run a small D_attack_dev agent sweep.
15. In parallel or next, implement continuous role-region steering as the main alternative.
```

---

# 19. Main differences from soft-pairwise v1

Old:

\[
w_r=\max(0,p_r-\tau_r)
\]

\[
\Delta h
=
\alpha\sum_r w_rv_{r\rightarrow T}.
\]

Problems:

```text
four independent false-positive gates
hard zeros on many genuinely suspicious tokens
token-level spikes
classifier probability treated as geometric distance
raw pair-vector norm determines intervention size
multiple role vectors can inflate/cancel magnitude
no explicit relative-norm cap
```

New:

\[
\boxed{
\text{smoothed standardized wrong-role evidence}
\rightarrow
\text{one joint clean gate}
\rightarrow
\text{soft role routing}
\rightarrow
\text{normalized role-space direction}
\rightarrow
\text{bounded relative intervention}
}
\]

This keeps the original discrete role-confusion hypothesis while removing the most obvious failure modes seen in the current implementation.
