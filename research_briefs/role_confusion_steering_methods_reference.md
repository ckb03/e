# Role-Confusion Steering: Methods and Analysis Reference

## Goal

Explore whether prompt injection caused by **role confusion** can be mitigated by activation interventions that make content be treated more like its **true architectural role** (e.g. tool output remains tool-like even when its text imitates a user/developer/assistant instruction).

Do **not** assume in advance that "role" is one steering vector. Treat the following as competing hypotheses:

1. **Absolute role direction**: each role has a fixed offset from a common neutral baseline.
2. **Source→target displacement**: correcting Tool←User differs from Tool←Developer.
3. **Continuous low-dimensional role state**: an activation can be partly Tool, partly User, etc.; correction should depend on its current position.
4. **Higher-dimensional / nonlinear state**: no fixed vector or small role subspace is enough.
5. **Role representation is decodable but not causally important**: probes can detect role, but changing it does not fix injection.

The experiments below are intended to distinguish these hypotheses.

---

# 1. Notation

For the same underlying text/content \(x_i\):

- \(h_i^r\): activation when placed under role \(r\)
- \(r \in \{\text{system, developer, user, assistant, tool}\}\) (adapt to model)
- \(\mu_r = \frac{1}{N}\sum_i h_i^r\): role centroid
- \(\mu_0\): optional no-tag / neutral baseline centroid
- \(h\): activation of the current attacked example
- \(t\): true architectural role
- \(s\): perceived/spoofed role
- \(\alpha\): steering strength
- \(B \in \mathbb{R}^{d\times k}\): orthonormal basis of a learned role subspace
- \(z(h)=B^\top(h-\mu)\): coordinates of activation \(h\) inside role space

Prefer collecting activations from **matched content under different tags**, because this removes much semantic variance.

Run everything layer-wise first. Do not assume the best layer.

---

# 2. Steering methods worth trying

## 2.1 Absolute role vector relative to neutral/no-tag baseline

Compute:

\[
v_r = \mu_r-\mu_0
\]

Intervene:

\[
h' = h+\alpha v_t
\]

### Assumption

A role is an approximately fixed additive feature relative to a common neutral state:

\[
h_r \approx h_{\text{content}}+v_r.
\]

### Why try it

It directly tests the intuitive idea of a standalone "Tool vector", "User vector", etc.

### Important weakness

If the attacked activation is already displaced toward another role, adding only \(v_t\) may **add the target without removing the wrong-role component**.

If role offsets are additive, then the ideal source→target correction is naturally

\[
v_t-v_s
=
(\mu_t-\mu_0)-(\mu_s-\mu_0)
=
\mu_t-\mu_s.
\]

So comparing absolute-role steering against pairwise correction is highly informative.

---

## 2.2 Paired mean-difference / CAA-style steering

For matched content:

\[
d_i^{s\rightarrow t}=h_i^t-h_i^s
\]

and

\[
v_{s\rightarrow t}
=
\frac1N\sum_i d_i^{s\rightarrow t}.
\]

Intervene:

\[
h'=h+\alpha v_{s\rightarrow t}.
\]

For example:

- Tool content spoofed as User: \(v=\mu_{\text{Tool}}-\mu_{\text{User}}\)
- User content spoofed as Developer: \(v=\mu_{\text{User}}-\mu_{\text{Developer}}\)

### Assumption

Changing perceived role \(s\rightarrow t\) is approximately a constant translation:

\[
h_t \approx h_s+v_{s\rightarrow t}.
\]

### Why it may beat absolute steering

It explicitly **adds the desired target component and removes the source component**.

### Key diagnostic

Check cosine similarity among individual differences:

\[
\cos(d_i,d_j).
\]

If pairwise differences are consistently aligned and SVD has one dominant singular value, a fixed pairwise vector is well justified.

---

## 2.3 Hard probe-selected pairwise steering

Train a multiclass linear role probe:

\[
p(r|h)=\operatorname{softmax}(Wh+b).
\]

For an attacked activation, infer:

\[
\hat s = \arg\max_r p(r|h).
\]

If true role is \(t\), apply:

\[
h'=h+\alpha v_{\hat s\rightarrow t}.
\]

Possible \(v_{\hat s\rightarrow t}\):

- mean difference
- LDA direction
- SVD-derived direction
- learned behavioral vector

### Assumption

The representation can be summarized as one discrete currently-perceived role, and the proper intervention depends on the **current wrong role**.

### Important limitation

An activation may not be cleanly "User" or "Tool". It may occupy an intermediate state. Hard argmax throws this information away.

This method is useful mainly as a baseline against continuous role-space correction.

---

## 2.4 Probe-direction steering

For binary \(A/B\):

\[
p(B|h)=\sigma(w^\top h+b)
\]

Use:

\[
v = \frac{w}{\|w\|}.
\]

For multiclass:

\[
v_{A\rightarrow B}=w_B-w_A.
\]

Intervene:

\[
h'=h+\alpha v.
\]

### Assumption

The direction most useful for **decoding** role is also causally capable of changing role-dependent behavior.

### Why important

This is a strong falsification test.

If probe steering changes predicted "Toolness" a lot but does not reduce attack success, then role is **decodable but the probe direction is not the causal mechanism**.

---

## 2.5 LDA / Fisher steering

For a pair of roles:

\[
S_W=\operatorname{Cov}(h^A)+\operatorname{Cov}(h^B)
\]

\[
v_{A\rightarrow B}
=
(S_W+\lambda I)^{-1}(\mu_B-\mu_A).
\]

Normalize if desired.

### Assumption

Useful role information lies in directions with:

- large between-role difference
- small within-role nuisance variance

LDA is effectively a **whitened centroid difference**.

### Why try it

Mean difference may be dominated by high-variance semantic directions. LDA downweights them.

For 5 roles, multiclass LDA yields at most \(5-1=4\) discriminant dimensions.

---

## 2.6 SVD/PCA direction on matched role differences

For one pair \(A\rightarrow B\), construct:

\[
D=
\begin{bmatrix}
(h_1^B-h_1^A)^\top\\
\vdots\\
(h_N^B-h_N^A)^\top
\end{bmatrix}
\]

and compute:

\[
D=U\Sigma V^\top.
\]

Use first right singular vector:

\[
v=V[:,1].
\]

Choose sign so:

\[
v^\top(\mu_B-\mu_A)>0.
\]

### Assumption

Individual role changes may have different magnitudes but share a common dominant direction:

\[
d_i\approx a_i v+\epsilon_i.
\]

This is weaker than assuming every example has the same displacement.

### What to inspect

The singular spectrum:

\[
\sigma_1,\sigma_2,\ldots
\]

If \(\sigma_1\) dominates, one vector is plausible.

If several singular values are large, use a subspace rather than one vector.

---

## 2.7 Low-rank role-subspace steering

Build a \(k\)-dimensional basis:

\[
B=[v_1,\ldots,v_k]
\]

from:

- SVD of matched role differences
- SVD of centered role centroids
- multiclass LDA
- another supervised subspace method

Then role information is represented by coordinates:

\[
z(h)=B^\top(h-\mu).
\]

### Simple subspace-restricted pairwise correction

\[
\Delta h
=
\alpha BB^\top(\mu_t-\mu_s).
\]

### Assumption

"Role" consists of multiple latent factors rather than one axis, but those factors live in a relatively low-dimensional linear subspace.

Examples might include:

- source identity
- authority / privilege
- model-vs-external speaker
- generated-vs-provided content

Do not name dimensions without independent validation.

---

## 2.8 Continuous adaptive target-role correction

This is a particularly important method.

Let current role coordinates be:

\[
z(h)=B^\top(h-\mu).
\]

For target role \(t\):

\[
z_t=B^\top(\mu_t-\mu).
\]

Then:

\[
\boxed{
\Delta h
=
\alpha B(z_t-z(h))
}
\]

and

\[
h'=h+\Delta h.
\]

Equivalent form:

\[
h'
=
h+\alpha BB^\top(\mu_t-h).
\]

### Interpretation

This **already subtracts the currently perceived role/location**.

It does not merely add \(z_t\).

- If activation is already Tool-like: correction is small.
- If it has moved strongly toward User: correction is larger.
- If it lies between roles: correction is continuous rather than choosing one argmax class.

### Assumption

Roles correspond to locations in a low-dimensional role space, and moving toward the clean target-role region causally restores correct treatment.

This is the cleanest continuous generalization of pairwise \(t-s\) steering.

---

## 2.9 Probe-calibrated steering strength

Even with a fixed direction, make strength depend on the amount of deviation.

Example:

\[
g(h,t)
=
1-p_{\text{probe}}(t|h)
\]

then:

\[
h'=h+\alpha g(h,t)v.
\]

Or use a margin:

\[
g(h,t)
=
\max(0,\max_{r\neq t}\ell_r(h)-\ell_t(h))
\]

where \(\ell_r\) are probe logits.

For adaptive role-space steering, the natural magnitude is already:

\[
\|z_t-z(h)\|.
\]

### Assumption

More severe role confusion needs stronger correction; clean inputs should be changed minimally.

### Important test

Compare:

- fixed \(\alpha\)
- probe-calibrated \(\alpha\)
- role-space-distance calibrated \(\alpha\)

Measure whether adaptive strength improves the ASR/utility Pareto frontier.

---

## 2.10 Project away the wrong/confusion direction

Instead of adding target role, remove the undesired component.

For normalized confusion direction \(u\):

\[
h'
=
h-\alpha uu^\top(h-\mu).
\]

For a subspace \(B_{\text{wrong}}\):

\[
h'
=
h-\alpha B_{\text{wrong}}B_{\text{wrong}}^\top(h-\mu).
\]

### Assumption

The vulnerability is caused by acquisition of an unwanted role component; simply removing that component may be enough.

### Useful comparison

For Tool←User spoofing compare:

1. \(+\) Tool
2. \(-\) User
3. \(+\) Tool \(-\) User
4. adaptive move from current state to Tool

These need not be behaviorally equivalent.

---

## 2.11 Behaviorally optimized single steering vector

Make \(v\) trainable and inject:

\[
h'=h+\alpha v.
\]

Optimize actual hierarchy-respecting behavior.

Conceptual loss:

\[
L(v)
=
L_{\text{correct hierarchy}}
+
\beta L_{\text{utility}}
+
\lambda\|v\|^2.
\]

For preference-style data:

\[
L
=
-\log p_v(y^+|x)
+
\gamma \log p_v(y^-|x)
+
\beta D_{\mathrm{KL}}(p_v||p_0).
\]

Where:

- \(y^+\): completion respecting true instruction hierarchy
- \(y^-\): injection-following completion
- KL / utility term prevents destroying normal capability

Initialize with:

- zero
- mean difference
- LDA
- SVD PC1

### Assumption

A useful additive causal direction exists, but statistical extraction methods do not necessarily find it.

### Important scientific comparison

If learned \(v\) strongly beats mean difference, then the causal behavior direction is not simply the largest natural role displacement.

---

## 2.12 Behaviorally optimized low-rank / subspace intervention

Learn a basis \(B\) and role coordinates \(a_r\):

\[
v_r=Ba_r
\]

so:

\[
v_{s\rightarrow t}=B(a_t-a_s).
\]

Or learn the adaptive correction directly.

### Assumption

Role is multidimensional, but still approximately low rank.

### Advantage

For 5 roles, this shares structure across all role pairs rather than learning 20 independent directed vectors.

---

## 2.13 Low-rank affine transformation

Allow a state-dependent linear map:

\[
h'
=
h+Ah+b
\]

with low-rank:

\[
A=UV^\top,\qquad U,V\in\mathbb R^{d\times k}.
\]

Train using injection/hierarchy loss plus a strong utility regularizer.

### Assumption

Role correction requires rotations, scaling, or state-dependent transformation, not only translation.

### Interpretation

If this beats all additive/subspace methods substantially, a fixed "steering vector" is probably too restrictive.

---

## 2.14 Nonlinear conditional correction

Learn:

\[
\Delta h=f_\phi(h,t)
\]

with a small MLP / adapter constrained to low capacity.

### Assumption

The correction is strongly content/state dependent and cannot be represented by one linear map.

Use only after simpler hypotheses fail; interpretability drops quickly.

---

## 2.15 Activation patching oracle — not a deployable method, but essential

For a spoofed example and a matched clean counterpart, replace:

\[
h_{\text{spoof}}
\rightarrow
h_{\text{clean}}
\]

at selected token(s), layer(s), or spans.

### Why this is critical

It gives an approximate **causal upper bound** for activation steering at that location.

Interpretation:

- If full clean patching does not fix behavior, searching for a better vector at that site is probably pointless.
- If clean patching works but role-vector steering fails, the relevant correction exists in the activation but is not captured by the chosen low-dimensional representation.
- If low-rank correction approaches patching performance, strong evidence that role confusion is captured by that subspace.

This should be run very early.

---

## 2.16 Attention/value priority steering (V-Steer-like baseline)

Instead of treating role as a residual-stream state, identify attention heads where low-priority content has excessive causal influence over high-priority content.

Intervene on cached values / attention contribution to:

- boost high-priority span influence
- suppress low-priority span influence

### Assumption

Prompt injection is fundamentally a **priority-routing / causal-influence problem**, not a role-representation problem.

### Why include it

If this strongly beats role-space steering, it suggests role probes may be correlates of the failure rather than its causal bottleneck.

---

# 3. Highest-information experiment: absolute vs pairwise vs continuous vs oracle

For every spoof pattern, e.g.:

- true Tool, User-like content
- true Tool, Developer-like content
- true User, Developer-like content
- true User, Assistant-like content

compare:

### A. Absolute target-role steering

\[
\Delta h=\alpha v_t
\]

### B. Hard source→target steering

\[
\Delta h=\alpha(\mu_t-\mu_{\hat s})
\]

where \(\hat s\) comes from the probe.

### C. Continuous role-space correction

\[
\Delta h
=
\alpha B[z_t-z(h)]
\]

### D. Full clean activation patch

\[
h_{\text{spoof}}\rightarrow h_{\text{clean}}
\]

Interpret:

- **A works** → an absolute target feature may be sufficient.
- **B > A** → subtracting the wrong-role component matters.
- **C > B** → role confusion is better modeled continuously / multidimensionally than as one discrete wrong class.
- **D >> C** → low-dimensional role representation misses causally relevant structure.
- **D fails** → this layer/token/site is probably not a useful causal intervention point.

This experiment should strongly determine what to investigate next.

---

# 4. PCA / SVD / LDA / geometric analyses

These analyses are not just visualization. Each tests assumptions behind particular steering methods.

---

## 4.1 Raw PCA — visualization only, weak evidence

Stack raw activations:

\[
X=[h_1^\top;\ldots;h_N^\top]
\]

PCA:

\[
X_c=U\Sigma V^\top.
\]

Plot PC1 vs PC2 colored by role.

### Problem

Largest activation variance may reflect:

- token identity
- semantics
- position
- punctuation
- template
- length

rather than role.

Useful as a sanity visualization, but do not use raw PCA separation alone as evidence of a role mechanism.

---

## 4.2 Within-content-centered PCA/SVD

For each matched content item \(i\):

\[
\bar h_i=\frac{1}{R}\sum_r h_i^r
\]

\[
z_i^r=h_i^r-\bar h_i.
\]

Stack all \(z_i^r\) and do SVD/PCA.

### What this isolates

Variation caused by **changing role while holding content fixed**.

### Outputs

- PC1/PC2 scatter colored by role
- singular-value scree plot
- role centroid coordinates in PC space

This is much more useful than PCA on raw activations.

---

## 4.3 Role-centroid SVD

Using centered role centroids:

\[
\tilde\mu_r=\mu_r-\mu
\]

construct:

\[
M=
\begin{bmatrix}
\tilde\mu_1^\top\\
\cdots\\
\tilde\mu_R^\top
\end{bmatrix}.
\]

Compute:

\[
M=U\Sigma V^\top.
\]

For \(R=5\), rank is at most \(R-1=4\).

### Question answered

**How many dimensions are needed to represent the geometry between the role centroids?**

Possible interpretations:

- one dominant singular value → near-1D role geometry
- two dominant values → approximately 2D
- four comparable values → role centroid geometry is genuinely multidimensional

Use this to choose \(k\) for role-subspace steering.

---

## 4.4 SVD of matched pairwise difference vectors

For pair \(A\rightarrow B\):

\[
D_{AB}
=
[h_1^B-h_1^A;\ldots;h_N^B-h_N^A].
\]

SVD:

\[
D_{AB}=U\Sigma V^\top.
\]

### Questions answered

1. Are individual source→target changes approximately parallel?
2. Is one steering vector reasonable?
3. How many dimensions are needed?

Useful statistics:

- explained variance ratio
- \(\sigma_1/\sum_j\sigma_j\)
- pairwise cosine distribution between \(d_i\)
- reconstruction error using top \(k\) PCs

---

## 4.5 SVD of all pairwise role vectors

Construct role-pair centroid differences:

\[
d_{r,s}=\mu_r-\mu_s.
\]

Stack all pairs in matrix \(D\) and perform SVD.

### Question answered

Are apparently different pairwise role vectors combinations of a few shared axes?

Example possible result:

- one axis separates trusted instruction sources from data/output
- another separates model-generated from externally supplied text

Do not assign semantic labels until independently tested.

---

## 4.6 Multiclass LDA

Compute between-class and within-class covariance:

\[
S_B,\quad S_W.
\]

Solve:

\[
S_Bv=\lambda S_Wv.
\]

For 5 roles, at most 4 discriminant directions.

### Question answered

Which dimensions best separate known roles **relative to within-role variability**?

### Visualization

Plot:

- LD1 vs LD2
- LD1 vs LD3
- role centroids + confidence ellipses if useful

Compare LDA subspace against PCA/SVD subspace.

---

## 4.7 Linear-probe analysis

Train probe only on held-out matched-content data.

Measure:

- role accuracy
- confusion matrix
- AUROC / log loss if useful
- calibration
- cross-template generalization
- cross-content generalization
- attack examples without retraining

### Most important mechanistic measurement

For a spoofed example compare:

\[
p_{\text{probe}}(r|h)
\]

before and after intervention against actual attack behavior.

Plot:

\[
\Delta \text{probe target-role score}
\quad \text{vs}\quad
\Delta \text{attack success}.
\]

If they decouple, probe geometry is not necessarily causal.

---

## 4.8 Role-space trajectory through the token sequence

For each token:

\[
z_t=B^\top(h_t-\mu).
\]

Plot trajectory through:

- clean Tool text
- Tool text containing User-like injection
- same attacked text after steering

### Question answered

Does the model progressively drift from the true role toward the spoofed role?

Does steering prevent or reverse that trajectory?

This can be one of the strongest visual explanations of role confusion.

---

## 4.9 Distance to role centroids

In role space:

\[
d_r(h)=\|z(h)-z_r\|.
\]

Better: use covariance-aware Mahalanobis distance:

\[
d_r^2(h)
=
(z-z_r)^\top\Sigma_r^{-1}(z-z_r).
\]

### Uses

- quantify amount of role confusion
- choose adaptive steering magnitude
- detect whether attacks lie between roles rather than at one centroid
- compare clean vs spoofed distributions

---

## 4.10 Principal angles between role subspaces across layers

Estimate role basis \(B^{(\ell)}\) at each layer.

Compute principal angles between:

\[
\operatorname{span}(B^{(\ell_1)})
\quad\text{and}\quad
\operatorname{span}(B^{(\ell_2)}).
\]

### Question answered

Where does a stable role subspace emerge?

This can guide which layers to steer.

Also compare:

- tag-induced role subspace
- style-induced role subspace

If they align strongly, it supports the idea that architectural role tags and spoofed role style use overlapping internal geometry.

---

## 4.11 Cross-role vector algebra / consistency tests

If roles are additive offsets from a common baseline, then approximately:

\[
v_{A\rightarrow C}
\approx
v_{A\rightarrow B}+v_{B\rightarrow C}.
\]

Test:

\[
\epsilon_{ABC}
=
\|v_{A\rightarrow C}-(v_{A\rightarrow B}+v_{B\rightarrow C})\|.
\]

Also test antisymmetry:

\[
v_{A\rightarrow B}\approx-v_{B\rightarrow A}.
\]

### Why useful

These directly test whether role geometry behaves like a coherent linear coordinate system.

If badly violated, fixed pairwise vectors are less principled.

---

## 4.12 Generalization of steering vectors across content

Train/extract vectors on content set \(X\), evaluate on disjoint semantic domains \(Y\).

Examples:

- neutral prose
- instructions
- code
- webpages
- retrieved documents
- tool outputs

### Question answered

Is the steering direction truly about role, or does it exploit semantic/style shortcuts?

This is essential.

---

# 5. Causal analyses

## 5.1 Layer sweep

For every candidate method, steer one layer at a time.

Measure:

- attack success rate (ASR)
- hierarchy-following accuracy
- benign task utility
- KL / output distribution shift

Plot intervention effect by layer.

The best decoding layer need not be the best causal layer.

---

## 5.2 Token-position / span sweep

Intervene on:

- all content tokens
- last token only
- injection tokens only
- full low-priority span
- boundary/tag token
- selected late tokens

This helps identify whether role state is:

- persistent across span
- accumulated token-by-token
- encoded mainly at boundaries
- only causally important near generation

---

## 5.3 Dose-response curve

For each steering direction test:

\[
\alpha \in \{-4,-2,-1,-0.5,0,0.5,1,2,4\}
\]

(or norm-calibrated equivalents).

Plot:

- ASR vs \(\alpha\)
- benign utility vs \(\alpha\)
- probe role score vs \(\alpha\)

### What good causal evidence looks like

A smooth dose-response:

\[
\alpha\uparrow
\Rightarrow
\text{target role score}\uparrow
\Rightarrow
\text{ASR}\downarrow
\]

until oversteering begins to harm utility.

---

## 5.4 Reverse-direction control

If \(v_{U\rightarrow T}\) improves Tool robustness, test:

\[
-v_{U\rightarrow T}.
\]

A causal direction should often worsen or reverse the effect.

This is stronger evidence than only showing one direction helps.

---

## 5.5 Random-direction and norm-matched controls

For each intervention vector \(v\), compare with random vectors of equal norm at the same layer.

This controls for generic perturbation / activation magnitude effects.

---

## 5.6 Orthogonal-component control

If role subspace is \(B\), construct a random direction orthogonal to it:

\[
v_\perp=(I-BB^\top)q.
\]

Norm-match to steering vector.

If role-space steering strongly beats \(v_\perp\), that supports specificity.

---

# 6. Measurements

Do not select methods only by probe score.

Primary behavioral metrics:

1. **Attack success rate (ASR)** — lower is better.
2. **True instruction / hierarchy adherence** — higher is better.
3. **Benign task success / utility** — should remain high.
4. **Over-refusal / over-suppression** — should remain low.
5. **KL or output-distribution shift** on benign data — lower is better.
6. **Role-probe movement** — mechanistic secondary metric, not final objective.

A useful comparison is the Pareto frontier:

\[
\text{security improvement}
\quad \text{vs} \quad
\text{utility degradation}.
\]

Also record compute and complexity.

---

# 7. Recommended experiment order

## Stage 0 — causal feasibility

Before searching many vectors:

1. Build matched clean/spoofed examples.
2. Perform **clean activation patching** across layers/tokens.
3. Find locations where patching materially reduces ASR.

If no causal effect exists at a location, do not spend time optimizing steering vectors there.

---

## Stage 1 — cheap steering baselines

At promising layers compare:

1. absolute target vector \(v_t=\mu_t-\mu_0\)
2. pairwise mean difference \(\mu_t-\mu_s\)
3. probe direction \(w_t-w_s\)
4. LDA direction
5. SVD PC1 of matched differences
6. projection away from wrong role

Run a shared \(\alpha\) sweep.

This is cheap and highly informative.

---

## Stage 2 — determine dimensionality

Run:

- within-content-centered PCA/SVD
- role-centroid SVD
- pairwise-difference SVD
- multiclass LDA
- cosine consistency
- cross-layer principal angles

Decide whether role geometry is approximately:

- 1D
- low-dimensional \(k>1\)
- not well described by a stable linear subspace

---

## Stage 3 — continuous adaptive steering

If low-dimensional structure exists, compare:

### Hard pairwise

\[
\Delta h=\alpha(\mu_t-\mu_{\hat s})
\]

against

### Continuous role-space correction

\[
\Delta h
=
\alpha B[z_t-z(h)].
\]

This directly tests whether the amount/direction of intervention should depend on the current activation state.

---

## Stage 4 — learn intervention from behavior

If statistical vectors help but are not optimal:

1. behaviorally optimize a single vector
2. behaviorally optimize a low-rank role basis
3. low-rank affine map
4. only then consider nonlinear correction

Always regularize against benign utility loss.

---

## Stage 5 — compare mechanism classes

Compare best residual-stream role steering against:

- attention/value priority steering (V-Steer-like)
- training-based hierarchy interventions if available

This determines whether role representation is the most useful causal handle.

---

# 8. Suggested core ablation table

For each true-role / spoof-role pair:

| Method | Uses current perceived state? | Dimensionality | Learned from behavior? |
|---|---:|---:|---:|
| Absolute \(+v_t\) | No | 1 | No |
| Pairwise \(\mu_t-\mu_s\) | Hard discrete \(s\) | 1 | No |
| Probe \(w_t-w_s\) | Hard discrete \(s\) | 1 | No |
| LDA | Hard discrete \(s\) | 1 | No |
| Difference-SVD PC1 | Hard discrete \(s\) | 1 | No |
| Projection away from \(s\) | Yes / hard | 1 | No |
| Continuous role-space correction | Yes / continuous | \(k\) | No |
| Learned vector | Optional | 1 | Yes |
| Learned low-rank map | Yes | \(k\) | Yes |
| Full clean activation patch | Exact matched state | Full | Oracle |
| Priority/attention steering | Different mechanism | N/A | Possibly |

Report:

- ASR
- hierarchy accuracy
- benign utility
- KL
- role-probe score before/after
- intervention norm

---

# 9. Most important hypotheses to test

## H1: Absolute role feature

\[
h_r\approx h_{\text{content}}+v_r.
\]

Prediction: absolute \(+v_t\) works nearly as well as pairwise correction.

---

## H2: Pairwise source→target displacement

\[
h_t-h_s\approx v_{s\rightarrow t}.
\]

Prediction: pairwise steering beats absolute target steering; matched differences are highly aligned.

---

## H3: Continuous role space

\[
h_{\text{role}}\approx Bz.
\]

Prediction: several singular values matter, and

\[
B[z_t-z(h)]
\]

beats hard pairwise correction.

---

## H4: State-dependent linear transformation

Prediction: low-rank affine intervention materially beats all additive/subspace translations.

---

## H5: Role is decodable but not causal

Prediction:

- probe accuracy is high
- steering changes probe score
- behavioral ASR barely changes
- clean activation patching or priority steering may still work

This is scientifically important, not a failed experiment.

---

# 10. A useful interpretation rule

Do **not** conclude that a direction is a "role vector" merely because:

- PCA separates roles
- a probe classifies roles
- the direction changes probe logits

Stronger evidence is:

1. matched-content role changes align with the direction;
2. the direction generalizes across semantic domains;
3. intervention has a monotonic causal effect on hierarchy behavior;
4. reverse steering reverses the behavioral effect;
5. norm-matched random/orthogonal directions do not;
6. effect survives held-out attack styles;
7. intervention preserves benign utility.

The strongest mechanistic story would be:

> Spoofed content moves activations away from the externally correct role region toward another role region; the magnitude of that displacement predicts attack success; causally moving the activation back toward the correct role region reduces attack success; and the effect is specific to the learned role subspace.

---

# 11. Minimal implementation priority for Codex

If time/compute is limited, implement these first:

1. **Matched activation collection** across roles.
2. **Layer-wise clean activation patching oracle**.
3. **Pairwise mean-difference steering**.
4. **Absolute no-tag→role steering**.
5. **Multiclass linear probe + hard pairwise steering**.
6. **Difference SVD + singular spectrum**.
7. **Within-content-centered role SVD/PCA**.
8. **Continuous low-rank target-centroid correction**:
   \[
   h'=h+\alpha B[z_t-z(h)].
   \]
9. **Dose-response + reverse/random controls**.
10. If promising, **behaviorally optimize vector / low-rank map**.

This sequence should reveal whether it is worth investing in more sophisticated steering before doing expensive optimization.

---

# 12. Relevant method families / papers to inspect

- **Contrastive Activation Addition (CAA)** — mean contrastive activation differences used for steering.
- **Representation Engineering / RepE** — broader framework for extracting and manipulating representation directions.
- **BiPO** — behavior/preference-optimized steering instead of relying purely on activation statistics.
- **Conceptors for semantic steering** — multidimensional/subspace representation rather than a single concept vector.
- **LinEAS / affine activation steering** — learn more expressive linear/affine transformations.
- **Prompt Injection as Role Confusion** — matched-role probing and evidence that role-like style can shift internal role representation.
- **V-Steer** — prompt-injection mitigation by changing relative causal influence of high- vs low-priority spans rather than directly steering a role vector.

Use the papers as method inspiration, but evaluate every method using the same held-out attack and utility suite.

---

## Bottom line

The key experimental question is not:

> "What is the Tool vector?"

It is:

> **Given a known true architectural role \(t\) and a possibly confused activation \(h\), what is the lowest-complexity transformation \(F(h,t)\) that restores correct hierarchy behavior while minimally changing everything else?**

Test increasingly expressive hypotheses:

\[
\boxed{
+v_t
\;\rightarrow\;
v_t-v_s
\;\rightarrow\;
B[z_t-z(h)]
\;\rightarrow\;
\text{learned low-rank map}
\;\rightarrow\;
\text{nonlinear correction}
}
\]

and use clean activation patching as the causal reference point.
