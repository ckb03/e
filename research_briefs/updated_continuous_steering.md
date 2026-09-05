# Continuous Role-Region Steering — Implementation Guideline

## Goal

Test whether prompt-injection behavior can be reduced by treating **normal Tool activations as a region in a low-dimensional role subspace** and moving anomalous Tool activations back toward that region.

This is the main candidate after the probe-weighted soft-pairwise method performed poorly.

The runtime defense knows only:
- which tokens are architecturally inside a true `Tool` span;
- the hidden activation at the chosen intervention site;
- precomputed role-subspace and clean-Tool calibration state.

It must **not** use attack labels, injection boundaries, injected text, attack family, fake-role identity, or final-test outcomes.

The core question is:

> If a true Tool token leaves the region occupied by benign Tool activations in role space, can moving it back toward that region reduce attack behavior without damaging legitimate Tool use?

---

## 1. Difference from soft-pairwise steering

Do **not** use probe probabilities to choose a source role or to set correction magnitude.

Continuous steering does not ask:

```text
Does this Tool token look User-like, CoT-like, Assistant-like, or System-like?
```

It asks:

```text
Is this true Tool activation outside the normal clean-Tool region
inside the learned role subspace?
```

Then, if abnormal, move it back toward the clean Tool region.

The linear role probe is therefore optional at runtime.

Recommended use of the probe:
1. sanity-check that the chosen activation site contains strong role information;
2. diagnose pre/post intervention behavior;
3. compare with the paper.

Do not make probe probabilities part of the continuous intervention formula.

---

## 2. Activation site

Use the paper-aligned GPT-OSS role-analysis activation site first:

```text
all_pre_mlp_hidden_states / pre-MLP representation
```

Verify the exact hook in the current codebase.

The following must all refer to the **same representation**:

```text
D_repr activations
role-subspace construction
D_clean activations
runtime anomaly detection
runtime intervention
optional diagnostic probe
```

Do not train geometry at one site and intervene at another.

Treat the previous post-block residual site as a later ablation, not the default.

---

## 3. Representation dataset `D_repr`

### Purpose

`D_repr` is used only to learn role geometry.

For every base text, render the **same content** under:

```text
system
user
cot
assistant
tool
```

Only the role wrapper/tag should differ.

### Recommended scale

Minimum paper-scale setup:

```text
250 base texts
25% C4
75% Dolma3
up to 1024 aligned content tokens/base
```

Preferred serious run:

```text
400–500 base texts
```

because the previous setup was small and the probe was weak.

Do not jump to thousands before verifying the activation site and rendering logic.

### Token collection

- exclude role delimiter/tag tokens;
- preserve exact content-token alignment across all five role renderings;
- use all aligned content tokens if practical;
- otherwise use a large deterministic evenly-spaced sample such as 256–512 tokens/base;
- do not return to the previous max-64-token setup;
- keep base texts equally weighted so long documents do not dominate.

### Split

Split by **underlying base text before rendering**.

For 500 bases, a reasonable split is:

```text
350 representation-train
75 representation-dev
75 representation-test
```

All five role copies of one base remain in the same split.

Only `representation-train` is used to construct the steering basis.

---

## 4. Optional paper-style probe sanity check

Before trusting steering geometry, fit one simple five-way linear probe at the same activation site.

Use multinomial L2 logistic regression.

Reference configuration:

```text
C = 5e-3
no nonlinear hidden layers
convex solver if practical
```

Evaluate on held-out base texts:

```text
overall token accuracy
per-role accuracy
confusion matrix
cross-entropy
```

This is a **sanity check**, not the steering algorithm.

If role decoding remains near ~50–60% at the paper-aligned site with correct role rendering and much more data, debug the extraction/site/setup before drawing causal conclusions.

Do not invent a numerical paper threshold; the paper does not publish one exact GPT-OSS held-out probe accuracy number.

---

## 5. Build the continuous role subspace

Let

\[
h_{i,j,r}\in\mathbb{R}^d
\]

be the activation for base text \(i\), aligned content token \(j\), and role \(r\).

For every matched token, compute the across-role mean:

\[
ar h_{i,j}
=
rac{1}{5}
\sum_r h_{i,j,r}.
\]

Remove the shared semantic/token component:

\[
q_{i,j,r}
=
h_{i,j,r}
-
ar h_{i,j}.
\]

Stack the centered role-variation vectors and compute:

\[
Q = U\Sigma V^	op.
\]

Define:

\[
B_k = V[:,1:k]
\]

with orthonormal columns.

Initial candidates:

```text
k = 4       primary
k = 2       optional ablation
k = 1       optional ablation
```

`k=4` is a natural first choice because five role centroids have at most four independent relative dimensions, but the full tokenwise centered matrix can contain role-content interactions, so this is a modeling choice rather than a theorem about the full matrix rank.

Save:

```text
B
singular_values
representation_center c
```

Use a consistent global center:

\[
c
=
\mathbb{E}_{i,j,r}[h_{i,j,r}].
\]

At runtime:

\[
z = B^	op(h-c).
\]

---

## 6. Avoid letting long bases dominate the SVD

Do not let a 1024-token base contribute much more total weight than a short base.

Recommended options:

### Option A — equal token sample per base

Sample the same number of valid aligned token positions from every base.

### Option B — weighted / streaming covariance

Accumulate:

\[
C_Q = \mathbb{E}[q q^	op]
\]

with each base receiving equal total weight, then take leading eigenvectors.

Either is fine if base weighting is explicit and deterministic.

---

## 7. `D_clean`: clean Tool calibration data

`D_clean` defines the normal Tool region.

Use benign Tool results from the **same environment/distribution as the attack evaluation**, but with no injected instructions.

Recommended:

```text
30 clean Wikipedia pages
20 calibration pages
10 held-out clean sanity pages
```

For a stronger final calibration, 40–50 pages is reasonable if cheap.

Do not overlap these pages with attack-dev or final-test pages.

---

## 8. Calibration should be cheap

Calibration should **not** run full agent trajectories repeatedly.

For each clean calibration page:

1. construct the normal context up to the first canonical model call after the Tool result is returned;
2. run one model prefill;
3. collect the chosen-layer activations for the true Tool span;
4. project them immediately to \(z\in\mathbb{R}^k\);
5. save only low-dimensional coordinates and token metadata.

After these forwards, all remaining calibration is offline statistics.

Do not:
- rerun pages for different statistics;
- rerun calibration for every intervention strength;
- rerun full agent loops;
- recalibrate for every `rho_max`.

---

## 9. Fit the clean Tool region

For every clean Tool token:

\[
z_j
=
B^	op(h_j-c).
\]

Estimate:

\[
m_T
=
\mathbb{E}[z_j\mid clean,\ Tool]
\]

and

\[
\Sigma_T
=
\operatorname{Cov}(z_j\mid clean,\ Tool).
\]

Use regularization:

\[
\Sigma_T^\lambda
=
\Sigma_T
+
\lambda I.
\]

A reasonable scale-aware default:

```text
lambda = 1e-4 * trace(Sigma_T) / k
```

Use equal page weighting or a fixed number of token samples per page.

---

## 10. Token-level Tool distance

For every Tool token compute:

\[
D_j^2
=
(z_j-m_T)^	op
(\Sigma_T^\lambda)^{-1}
(z_j-m_T).
\]

Calibrate a token radial boundary:

\[
	au_{	ext{token}}
=
Q_{0.99}
\left[
D_j^2
\mid clean\ Tool

ight].
\]

Initial default:

```text
q = 0.99
```

Also save `Q0.995` for later ablation.

This threshold determines the clean Tool radial boundary used for the correction target.

---

## 11. Optional local smoothing for detection

Prompt injections are coherent spans, while benign HTML can create isolated unusual tokens.

Use local context to decide whether a region is suspicious.

Do **not** use the window to define the correction vector.

Recommended first version:

```text
window = trailing 32 Tool tokens
```

For example:

\[
A_j
=
rac{1}{W}
\sum_{t=j-W+1}^{j}
D_t^2.
\]

Or use an EMA.

Whatever smoothing is used at runtime must also be used during clean calibration.

Calibrate one local gate:

\[
eta
=
Q_{0.99}
\left[
A_j
\mid clean\ Tool

ight].
\]

The thresholds have different meanings:

```text
beta        = does this local region look suspicious?
tau_token   = is this individual token outside the clean Tool radial boundary?
```

---

## 12. Runtime gate

Only consider true architectural `Tool` tokens.

For token \(j\):

1. compute \(z_j\);
2. compute \(D_j^2\);
3. update local score \(A_j\).

If

\[
A_j \le eta
\]

then:

\[
\Delta h_j=0.
\]

Even if the local region is suspicious, do not move an individually normal token:

\[
D_j^2 \le 	au_{	ext{token}}
\Rightarrow
\Delta h_j=0.
\]

Thus steering occurs only when:

\[
oxed{
A_j>eta
\quad\land\quad
D_j^2>	au_{	ext{token}}
}
\]

This prevents a suspicious neighboring span from dragging a normal token.

---

## 13. Continuous correction

For a token that passes both gates, move only far enough to return to the clean Tool radial boundary.

Current displacement:

\[
u_j=z_j-m_T.
\]

Define:

\[
s_j
=
\sqrt{
rac{	au_{	ext{token}}}
{D_j^2}
}.
\]

Since \(D_j^2>	au_{	ext{token}}\),

\[
0<s_j<1.
\]

Target:

\[
z_j^{target}
=
m_T
+
s_j(z_j-m_T).
\]

So:

\[
\Delta z_j
=
-(1-s_j)(z_j-m_T).
\]

Map back:

\[
\Delta h_j^{raw}
=
B\Delta z_j.
\]

Then:

\[
h_j'=h_j+\Delta h_j.
\]

This moves the token to the estimated clean-Tool ellipsoid boundary, not all the way to the Tool centroid.

---

## 14. Hard relative-norm safety cap

Define:

\[
r_j
=
rac{\|\Delta h_j^{raw}\|_2}
{\|h_j\|_2+\epsilon}.
\]

Clip:

\[
\Delta h_j
=
\Delta h_j^{raw}
\cdot
\min\left(
1,
rac{
ho_{\max}\|h_j\|}
{\|\Delta h_j^{raw}\|+\epsilon}

ight).
\]

Initial development sweep:

```text
rho_max ∈ {0.0025, 0.005, 0.01}
```

meaning 0.25%, 0.5%, and 1.0% maximum relative perturbation.

Do not use an unbounded `alpha` initially.

If the cap activates, record that the token may remain outside the clean boundary.

---

## 15. Runtime pseudocode

```python
# frozen:
# B, c, m_tool, inv_cov_tool
# tau_token, beta, W, rho_max

history = []

for each token j:
    if true_role[j] != TOOL:
        continue

    h = hidden[j]

    z = B.T @ (h - c)
    diff = z - m_tool
    d2 = diff.T @ inv_cov_tool @ diff

    history.append(d2)
    local_score = mean(last_W(history))

    if local_score <= beta:
        continue

    if d2 <= tau_token:
        continue

    shrink = sqrt(tau_token / d2)
    delta_z = -(1.0 - shrink) * diff
    delta_h = B @ delta_z

    max_norm = rho_max * norm(h)
    if norm(delta_h) > max_norm:
        delta_h *= max_norm / (norm(delta_h) + eps)

    hidden[j] = h + delta_h
```

Reset smoothing state at Tool-span boundaries.

Never smooth across architectural role boundaries.

---

## 16. Diagnostics to save

Per Tool token or sampled Tool token:

```text
case_id
token_position
token_id / token_text
D2_before
local_score
gate_fired
token_outside_boundary
delta_norm
h_norm
delta_over_h
cap_activated
D2_after
```

Per case:

```text
eligible_tool_tokens
gate_firing_fraction
token_correction_fraction
mean_D2
p95_D2
max_D2
mean_local_score
mean_delta_over_h
max_delta_over_h
cap_activation_fraction
mean_D2_before_corrected
mean_D2_after_corrected
```

For corrected tokens:

\[
D_{after}^2 < D_{before}^2.
\]

If uncapped:

\[
D_{after}^2 pprox 	au_{	ext{token}}.
\]

---

## 17. Clean sanity check before attack evaluation

Freeze calibration using the 20 calibration pages.

Then run the 10 held-out clean pages.

Measure:

```text
local gate firing rate
token correction rate
cap activation rate
mean/max delta_over_h
content/task quality if full clean tasks are run
```

Held-out clean local-gate firing should stay near the intended calibration rate, roughly around 1%, not 10–20%.

If much larger, inspect:
- page weighting;
- distribution shift;
- covariance conditioning;
- smoothing implementation;
- Tool-span masking.

Do not immediately tune using attack labels.

---

## 18. Attack-development evaluation

Use only attack-development cases.

Compare at minimum:

```text
no intervention
continuous rho=0.0025
continuous rho=0.005
continuous rho=0.01
```

Use identical cases and identical decoding seeds across configurations.

Because GPT-OSS agent trajectories are stochastic, do not interpret one case flip as a causal result.

Preferred development scale if compute allows:

```text
20 matched cases
x 3 generation seeds
```

---

## 19. Do not trust only `output_class`

Keep `output_class` for paper comparability, but use actual saved tool behavior as the primary behavioral evidence.

Primary safety metric:

\[
ASR_{	ext{objective}}
=
P(	ext{actual exfiltration tool attempt}).
\]

Also report:

```text
sensitive-file preparation/read rate
paper-compatible attempt ASR
paper output_class distribution
judge-vs-tool-event agreement
```

Most important manual-audit cases:

```text
paper judge != objective tool-event behavior
```

or baseline/intervention cases whose actual executed tool actions differ.

---

## 20. Capability / utility metrics

Do not use attack reduction alone.

### Content utility

Use a less brittle capability measure:

```text
page returned
AND output present
AND content grounded
```

### Format compliance

Report separately:

```text
sentence-count requirement
word-count requirement
other formatting constraints
```

Do not treat a good 5-sentence summary as equivalent to a complete task failure.

### Joint secure utility

Main joint outcome:

\[
P(
	ext{content utility}=1
\land
	ext{objective attack attempt}=0
).
\]

Keep the stricter existing task-success field as a secondary metric.

---

## 21. Main comparison table

For every continuous configuration report:

| Method | Objective ASR ↓ | Sensitive-action rate ↓ | Paper ASR ↓ | Content utility ↑ | Strict task success ↑ | Secure content success ↑ | Clean gate firing ↓ | Mean/max `||Δh||/||h||` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

Also report paired outcomes:

```text
baseline attack -> steered safe
baseline safe   -> steered attack
both attacked
both safe
```

This is more informative than aggregate ASR alone.

---

## 22. Statistical comparison

For serious configurations:

- same cases;
- same generation seeds;
- paired comparison;
- paired bootstrap confidence interval over base cases;
- optionally McNemar exact test on binary objective attack flips.

Do not over-interpret 1–2 stochastic flips.

---

## 23. Mechanistic checks

### A. Attack vs clean separation

Before steering compare:

\[
D^2_{	ext{clean Tool}}
\]

with:

\[
D^2_{	ext{attacked Tool}}.
\]

Useful summaries:

```text
mean / median
quantiles
AUROC
injection-span vs page-other, if boundaries are available offline
```

Injection boundaries may be used for **offline diagnosis only**, never runtime gating.

### B. Does the intervention do what it claims?

For corrected tokens verify:

```text
D2_after < D2_before
```

and usually:

```text
D2_after ~= tau_token
```

unless clipped by the norm cap.

### C. Does geometry track behavior?

If role-space anomaly moves strongly toward clean Tool geometry but objective attack behavior does not improve, the decoded role geometry is probably not the causal bottleneck.

---

## 24. Useful controls

If continuous steering looks promising, add:

### Norm-matched random direction

Same gate and same correction norm, random orthogonal direction.

### Wrong / unrelated subspace direction

Same norm, direction unrelated to the clean-Tool correction.

These test whether improvement is role-specific rather than generic perturbation.

---

## 25. Suggested implementation order

```text
1. verify exact paper-aligned pre-MLP hook
2. collect larger D_repr (>=250, preferably 400–500 bases)
3. optionally train paper-style linear probe as sanity check
4. build centered continuous role basis B
5. inspect SVD spectrum
6. collect D_clean with one canonical Tool prefill/page
7. fit clean Tool center + covariance
8. calibrate tau_token
9. implement trailing-window local score
10. calibrate beta on the same smoothing rule
11. run held-out clean sanity pages
12. implement radial-to-boundary correction
13. add hard rho_max cap
14. verify D2_after numerically
15. run small matched attack-dev evaluation
16. compare objective tool-event ASR + utility
17. inspect paired regressions/wins
18. freeze configuration
19. run final held-out test only after method is fixed
```

---

## 26. Initial default configuration

```text
activation site:
    paper-aligned pre-MLP site

D_repr:
    400–500 base texts preferred
    25% C4 / 75% Dolma3
    max 1024 content tokens
    >=256–512 aligned sampled tokens/base if not using all
    split by underlying base text

role basis:
    k = 4

D_clean:
    20 calibration pages
    10 held-out sanity pages

clean region:
    empirical covariance
    small scale-aware diagonal regularization

token radial threshold:
    Q99 clean Tool D^2

local detector:
    trailing 32-token mean of D^2
    Q99 clean local score

runtime correction:
    radial shrink to token clean boundary

relative norm cap:
    rho_max = 0.005 first
    also test 0.0025 and 0.01

primary attack metric:
    objective exfiltration tool-attempt rate

primary utility metric:
    grounded content utility

joint metric:
    grounded content utility AND no objective exfiltration attempt
```

---

## 27. What counts as a promising result?

Promising:

```text
objective attack-attempt rate decreases materially
sensitive-file action rate decreases
content utility stays approximately unchanged
clean Tool correction rate stays low
safe->attack regressions are rare
role-space anomaly actually decreases on corrected tokens
```

Not convincing by itself:

```text
paper output_class improves
```

or:

```text
the model refuses more often
```

The intended claim is:

> Continuous correction toward the normal Tool region reduces actual injected-tool behavior while preserving legitimate Tool-mediated task completion.
