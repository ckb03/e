# Role-Confusion Steering — Representation Extraction and Algorithms

## Goal

Implement and compare two runtime defenses for Tool-result prompt injection:

1. **Soft pairwise steering**
2. **Continuous role-space steering**

Both operate under the same runtime information constraint:

Known:

```text
true architectural role = Tool
current hidden activation h
```

Unknown:

```text
whether injection exists
where injection begins
which role/style the injected content imitates
```

Therefore the algorithm must decide whether/how much to steer from the activation itself.

---

# 1. Activation hook

Use one consistent residual-stream location.

Recommended:

> transformer block output / residual stream after block `l`, before entering block `l+1`

For each layer:

\[
h_{\ell,i,j,r}\in\mathbb{R}^d
\]

where:

- `i`: base text/example
- `j`: aligned content-token position
- `r`: role
- `l`: transformer layer

Use the exact same hook for:

- representation extraction
- probe training
- steering
- analysis

Do not use role-tag tokens when learning role representations.

Initially steer only Tool-result content tokens during prompt prefill.

---

# 2. Train the 5-way role probe

Needed for **soft pairwise**.

Continuous steering does not require it at runtime, although the probe remains useful diagnostically.

For every layer independently, train:

\[
p_\ell(r|h)
=
\operatorname{softmax}(W_\ell h+b_\ell)
\]

where:

\[
r\in\{
system,user,cot,assistant,tool
\}.
\]

## Training data

Use only `D_repr/repr_train`.

Each sample:

```text
input = content-token activation
label = wrapper role
```

Use:

- multinomial logistic regression
- L2 regularization
- balanced role classes

Do not split tokens independently. Splits are already by base text.

If there are many tokens, either:

- sample a fixed number of content tokens from every base text, e.g. 64–128, or
- weight every base sequence equally

to avoid long documents dominating.

## Validate

On `repr_test`, record per layer:

```text
probe accuracy
cross entropy
5×5 confusion matrix
```

High probe accuracy is useful but is NOT by itself a reason to select that layer for steering.

Optionally temperature-calibrate logits on `repr_dev`:

\[
p_\ell(r|h)
=
softmax((W_\ell h+b_\ell)/T_\ell)
\]

Store:

```text
probe_W[layer]
probe_b[layer]
probe_temperature[layer]
```

---

# 3. Compute pairwise steering vectors

Needed for soft pairwise.

For each ordered role pair:

\[
s\rightarrow t
\]

and each layer `l`, use matched copies of the same base text.

First compute one average difference per base sequence:

\[
d^{(\ell)}_{i,s\rightarrow t}
=
\frac{1}{T_i}
\sum_j
\left(
h^{t}_{\ell,i,j}
-
h^{s}_{\ell,i,j}
\right)
\]

Then average equally over base sequences:

\[
\boxed{
v^{(\ell)}_{s\rightarrow t}
=
\frac1N
\sum_i
d^{(\ell)}_{i,s\rightarrow t}
}
\]

This prevents long texts from receiving disproportionate weight.

For five roles, store:

```text
pair_vector[layer][source_role][target_role][hidden_dim]
```

Define:

\[
v_{t\rightarrow t}=0.
\]

For the current first experiment, runtime target is always:

```text
t = Tool
```

so only these source vectors are actually needed:

```text
system -> tool
user -> tool
cot -> tool
assistant -> tool
tool -> tool = 0
```

Still compute all pairs if cheap because they are useful for analysis.

---

# 4. Compute the continuous role subspace

Needed for continuous steering.

For every matched base sequence, aligned token, and layer:

\[
\bar h_{\ell,i,j}
=
\frac{1}{R}
\sum_r
h^r_{\ell,i,j}
\]

where:

\[
R=5.
\]

Subtract the per-content mean:

\[
q^r_{\ell,i,j}
=
h^r_{\ell,i,j}
-
\bar h_{\ell,i,j}.
\]

This removes much of the underlying semantic/token representation and isolates variation caused by role.

Stack all centered vectors for layer `l`:

\[
Q_\ell
=
\begin{bmatrix}
q_1^\top\\
q_2^\top\\
\vdots
\end{bmatrix}.
\]

Compute:

\[
Q_\ell=U_\ell\Sigma_\ell V_\ell^\top.
\]

Candidate role basis:

\[
\boxed{
B_{\ell,k}
=
V_\ell[:,1:k]
}
\]

with orthonormal columns.

Initially consider:

```text
k ∈ {1, 2, 4}
```

Since five role centroids have at most four independent relative dimensions, `k=4` is a natural upper baseline for role-centroid geometry.

Store singular values as well.

---

# 5. Cheap layer screening

Do not run expensive agent steering on every layer.

For every layer compute:

## A. Probe quality

From `repr_test`:

\[
Acc_\ell
\]

## B. Role-confusion signal on `D_layer`

Run undefended `D_layer` cases and collect Tool-token activations.

For soft-pairwise, compute:

\[
m^{probe}_\ell
=
1-p_\ell(tool|h).
\]

Compare distributions for:

```text
clean Tool tokens
attacked Tool tokens
```

Useful summary:

- mean difference
- AUROC clean vs attacked
- percentile separation

For continuous steering, temporarily fit the clean Tool region using `D_clean` and compute Tool-distance:

\[
d^2_{\ell}(h).
\]

Again compare clean vs attacked.

Keep roughly:

```text
top 3 candidate layers
```

that have:

- reasonable probe/role separation
- clear clean→attack representation shift

Do not simply choose the layer with highest probe accuracy.

---

# 6. Causal / steering layer selection

Using only the approximately 20 attack cases in `D_layer`, evaluate the top candidate layers.

Two options:

## Preferred if clean/attack activations align well

Perform clean activation patching:

\[
h_{\ell}^{attack}
\rightarrow
h_{\ell}^{clean}
\]

for matched Tool content/span where alignment is meaningful.

Measure behavior improvement.

A layer where clean patching strongly changes attack behavior is a strong causal candidate.

## Simpler general option

Run a very small steering sweep for each top candidate layer.

For example:

```text
alpha ∈ {0.5, 1.0}
```

using one baseline steering method, preferably pairwise mean-difference.

Measure:

```text
attack success
task success
obvious degradation
```

Select the layer with the best causal behavior effect.

This avoids a full all-layer sweep.

For a scientific comparison between soft-pairwise and continuous steering:

- first evaluate both at the same selected layer
- later optionally allow each method to choose its own nearby optimal layer

---

# 7. Calibrate soft-pairwise gating from `D_clean`

At the selected layer `l*`, run all clean Tool tokens through the probe:

\[
p(r|h).
\]

For every wrong role:

\[
r\neq Tool
\]

estimate a threshold:

\[
\tau_r
=
Q_q
\left[
p(r|h)
\mid
\text{clean Tool token}
\right]
\]

where `Q_q` can initially be:

```text
95th percentile
```

and later test 99th percentile if false positives are too high.

Store:

```text
tau_system
tau_user
tau_cot
tau_assistant
```

No Tool threshold is necessary for the basic weighted formulation.

---

# 8. Complete soft-pairwise runtime algorithm

For every Tool content token `j` at selected layer `l*`:

## Step 1 — obtain activation

\[
h_j
\]

## Step 2 — infer soft perceived role

\[
p_r
=
softmax((W h_j+b)/T)_r
\]

for:

```text
system
user
cot
assistant
tool
```

Optionally smooth probe probabilities over a short window within the Tool span.

Initial implementation: no smoothing first.

If needed later:

```text
window = 8 or 16 tokens
```

Never smooth across role boundaries.

## Step 3 — compute only abnormal wrong-role mass

For every wrong role:

\[
w_r
=
\max(0,p_r-\tau_r).
\]

For Tool:

\[
w_{Tool}=0.
\]

This means ordinary clean fluctuations below their normal clean threshold cause no intervention.

## Step 4 — combine source→Tool corrections

\[
\boxed{
\Delta h_j
=
\alpha
\sum_{r\neq Tool}
w_r
v_{r\rightarrow Tool}
}
\]

## Step 5 — steer

\[
\boxed{
h'_j=h_j+\Delta h_j
}
\]

Continue the model forward pass using `h'_j`.

### Runtime pseudocode

```python
def soft_pairwise_hook(h, token_role):
    # h: [seq, hidden] at selected layer

    p = calibrated_role_probe(h)

    delta = zeros_like(h)

    for j in tool_content_tokens(token_role):
        for r in ROLES:
            if r == TOOL:
                continue

            excess = max(
                0.0,
                p[j, r] - clean_threshold[r]
            )

            delta[j] += (
                excess
                * pair_vector[r, TOOL]
            )

    return h + alpha * delta
```

No injection label or spoof-role label is used.

---

# 9. Fit clean Tool region for continuous steering

At selected layer `l*` and candidate rank `k`, use the SVD basis:

\[
B=B_{\ell^*,k}.
\]

Choose a global activation center `c`, e.g.:

\[
c
=
E[h]
\]

over representation-training activations at this layer.

For every clean Tool token:

\[
z=B^\top(h-c).
\]

Estimate:

\[
m_T
=
E[z|\text{clean Tool}]
\]

and regularized covariance:

\[
\Sigma_T
=
Cov(z|\text{clean Tool})+\lambda I.
\]

Choose small covariance regularization, e.g. relative shrinkage or:

```text
lambda = 1e-4 ... 1e-2 × average variance
```

depending on numerical scale.

Compute clean Mahalanobis distances:

\[
d_T^2
=
(z-m_T)^\top
\Sigma_T^{-1}
(z-m_T).
\]

Set threshold:

\[
\tau_T
=
95\text{th percentile of clean }d_T^2
\]

initially.

Store:

```text
B
center
tool_role_mean
tool_role_cov_inv
tool_distance_threshold
```

---

# 10. Complete continuous runtime algorithm

For every Tool content token:

## Step 1 — project into role space

\[
z
=
B^\top(h-c).
\]

## Step 2 — measure deviation from normal Tool region

\[
diff=z-m_T.
\]

\[
\boxed{
d^2
=
diff^\top
\Sigma_T^{-1}
diff
}
\]

## Step 3 — gate

If:

\[
d^2\le\tau_T
\]

then:

\[
\Delta h=0.
\]

The token is inside the normal clean Tool region.

## Step 4 — if outside, move toward boundary/Tool region

A conservative correction is:

\[
\rho
=
1-\sqrt{\frac{\tau_T}{d^2}}.
\]

Then:

\[
\Delta z
=
-\rho(z-m_T).
\]

This moves the point approximately back toward the edge of the accepted Tool region instead of forcing it all the way to the centroid.

Apply global steering strength:

\[
\boxed{
\Delta h
=
\alpha B\Delta z
}
\]

and:

\[
\boxed{
h'=h+\Delta h.
}
\]

### Runtime pseudocode

```python
def continuous_role_hook(h, token_role):
    # h: [seq, hidden] at selected layer

    z = (h - center) @ B
    delta_z = zeros_like(z)

    for j in tool_content_tokens(token_role):
        diff = z[j] - tool_mean

        d2 = diff @ tool_cov_inv @ diff

        if d2 <= tool_threshold:
            continue

        rho = 1.0 - sqrt(tool_threshold / d2)

        delta_z[j] = -rho * diff

    delta_h = delta_z @ B.T

    return h + alpha * delta_h
```

Again, no attack label or spoof-role label is required.

---

# 11. Hyperparameter tuning

Use only `D_tune`.

## Soft pairwise

Initially tune only:

```text
alpha ∈ {0.5, 1.0, 2.0}
```

Keep threshold fixed from `D_clean`.

Only tune threshold/smoothing if the basic method is promising.

## Continuous

Choose `k` mainly from SVD spectrum first.

Initial candidates:

```text
k = 1
k = 4
```

If both differ materially, optionally try `k=2`.

Then tune:

```text
alpha ∈ {0.5, 1.0, 2.0}
```

Do not launch a large grid.

---

# 12. Development validation

After choosing one configuration for each method, freeze it and run once on `D_devval`.

Record at minimum:

```text
baseline ASR
steered ASR
task success / utility
number/fraction of Tool tokens steered
mean intervention norm
```

For soft pairwise also record:

```text
mean wrong-role excess mass
contribution from each source role
```

For continuous also record:

```text
fraction outside Tool region
mean Mahalanobis distance
mean correction magnitude
```

If performance collapses on `D_devval`, return to the development stage.

Do not touch final test yet.

---

# 13. Final evaluation

Once method, layer, and hyperparameters are frozen, run on the existing 200-case final manifest.

The algorithm receives only:

```text
Tool span metadata
hidden activations
precomputed steering state
```

No final-test labels influence steering.

Report:

```text
baseline ASR
soft-pairwise ASR
continuous ASR
task utility/capability
```

If compute permits, run multiple model-generation seeds for the final configuration.

---

# 14. Useful analysis to run almost for free

From `D_repr`, produce per layer:

## Pairwise-vector consistency

For each base sequence:

\[
d_i=h_i^{Tool}-h_i^{User}
\]

etc.

Measure:

```text
cosine(d_i, mean_d)
```

If these are tightly aligned, fixed pairwise steering is plausible.

## SVD spectrum

Plot:

\[
\sigma_1,\sigma_2,\ldots
\]

for within-content-centered role data.

Interpretation:

- dominant `sigma_1` -> near-1D role representation plausible
- several important singular values -> continuous low-rank method more plausible

## Probe vs behavior

Measure whether steering changes:

\[
p(Tool|h)
\]

and whether that correlates with ASR reduction.

If probe score moves strongly but behavior does not, decodable role representation may not be the causal bottleneck.

---

# 15. Recommended engineering order

Implement in this sequence:

```text
1. D_repr activation collector
2. layer-wise 5-way logistic probes
3. layer-wise pairwise mean-difference vectors
4. layer-wise centered SVD bases
5. D_clean Tool activation collector
6. cheap layer diagnostics
7. select ~3 candidate layers
8. small causal/steering layer test on 20 attacks
9. freeze selected layer
10. soft-pairwise runtime hook
11. continuous runtime hook
12. tune on 20 attacks
13. validate on separate 20 attacks
14. freeze method
15. run final 200-case eval
```

---

# 16. Core comparison

The most important first comparison is:

## Soft pairwise

\[
\boxed{
\Delta h
=
\alpha
\sum_{r\neq Tool}
\max(0,p_r-\tau_r)
v_{r\rightarrow Tool}
}
\]

Hypothesis:

> confusion can be represented as a soft mixture of discrete role directions.

## Continuous

\[
\boxed{
\Delta h
=
\alpha B\Delta z
}
\]

where `Delta z` moves the current role-space point back toward the normal Tool region.

Hypothesis:

> confusion is better represented as continuous displacement inside a low-dimensional role subspace.

Both are valid general runtime algorithms because neither requires knowing whether an attack occurred or what role the attacker intended to imitate.
