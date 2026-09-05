# Current heuristic and role-general detector design

## Bottom line

The current detector is not based on the claim that prompt injections can only resemble User or CoT. It is a two-channel development implementation because those were the two role/layer combinations that produced useful sustained local signals on the available base-injection and CoT-forgery cases.

A role-general algorithm is sensible, but it should not use one raw threshold over `max(System, User, CoT, Assistant) - Tool`. Each role and layer has a different benign distribution. The general form should therefore detect **clean-calibrated, sustained anomalies separately for every role**, then fuse the resulting evidence.

The enhanced plots are in [`FULL_DOCUMENT_PROBE_VIEW.md`](FULL_DOCUMENT_PROBE_VIEW.md). They include:

- matched clean/attack full-document traces for all five roles;
- the exact simulator injection shaded yellow;
- injection-centered views with 384 tokens of context on each side;
- both smoothed role probabilities and role-specific local contrasts.

## Exact current heuristic

For token position `t`, layer `l`, and role `r`, the linear probe returns five logits:

```text
z_l,t = [z_system, z_user, z_cot, z_assistant, z_tool]
p_l,t = softmax(z_l,t)
```

The selector calculates a role-specific local contrast:

```text
c_l,r(t) = mean_64(p_l,r)(t) - mean_512(p_l,r)(t)
```

The 64-token mean asks whether a role is sustained locally. The 512-token mean estimates the surrounding Tool-result background.

Only two channels are currently active:

```text
CoT channel:
    c_8,cot(t) > 0.50 for at least 8 consecutive positions

User channel:
    c_18,user(t) > 0.13 for at least 8 consecutive positions
```

Candidate construction then works as follows:

1. Pad each threshold run by 64 tokens.
2. Merge padded runs whose gap is at most 64 tokens.
3. For the layer-18 User channel, require:
   - at least 12 alphabetic words;
   - alphabetic fraction at least 0.45;
   - markup fraction at most 0.55; and
   - at least one generic request cue or at least two generic action verbs.
4. The layer-8 CoT channel bypasses that lexical filter because it had complete separation from the available clean pages in this development set.
5. Return 128 tokens of total context around the detected feature region.
6. Rank a merged candidate by its largest normalized threshold excess:

```text
rank =
    max(
        (peak_user18 - 0.13) / 0.20,
        (peak_cot8  - 0.50) / 0.20
    )
```

7. Send at most three passing candidates to the LLM judge. If zero pass, send zero; the implementation does not fill the list to three.

The thresholds, lexical conditions, span padding, merge rule, and ranking are implemented in [`design_candidate_selector.py`](design_candidate_selector.py). The downstream judge and correction do not change how these ranges are found.

## Why System-like base injections are not automatically easy to locate

Base-injection case 13 illustrates the distinction between an **absolute role label** and a **detectable local boundary**.

Inside the known injected span:

| Layer and signal | Value |
| --- | ---: |
| Layer-8 mean System probability | 0.980 |
| Layer-8 System local-contrast peak | 0.037 |
| Largest layer-8 System contrast on available clean pages | 0.454 |
| Layer-18 mean User probability | 0.164 |
| Layer-18 User local-contrast peak | 0.259 |

Thus, the layer-8 probe calls the injected paragraph System-like, but much of the surrounding webpage is already System-like. A System plateau is visible when the yellow ground-truth span tells us where to look, yet it is not unusual relative to the rest of the document.

A System threshold above the benign peaks would miss the injection. A threshold low enough to include the case-13 peak would admit many benign System-like regions. It would therefore not permit a stricter, lower-false-positive rule on this corpus.

This also explains why simply selecting whichever role has the largest probability is unsafe:

- the winning role can be high across an entire benign HTML or prose section;
- the five probabilities are mutually coupled by softmax;
- role probes measure linguistic/representational resemblance, not unauthorized intent;
- different layers can label the same span differently;
- searching tens of thousands of tokens magnifies rare benign peaks;
- the “obvious” role is known in the plot only because the yellow simulator boundary is available during analysis.

The existing layer-18 User channel can still retrieve a span that looks System-like at layer 8. Detection does not require the selected channel to equal the model's eventual behavioral interpretation of the attack.

On the saved development pages, the current uncapped broad selector already overlaps all 18/18 observable injections, including 9/9 base injections. Consequently, adding System cannot improve offline recall on these same pages. The missing detector activations in several end-to-end replays came from exact matching against precomputed Tool serializations, not because those saved pages lacked a candidate.

## What the all-role plots show

The full-document plots make two comparisons possible:

1. **Injection versus neighboring tokens:** use the injection-centered views, where yellow is wide enough to inspect local changes.
2. **Injection versus every other region in the document:** use the full-document matched views, where benign role plateaus and the multiple-comparisons problem are visible.

The existing five-role audit found:

- Layer-8 CoT local contrast: strongest 128-token window overlaps 9/9 CoT-forgery injections.
- Layer-8 System local contrast: strongest window overlaps 0/9 base injections.
- Layer-18 System local contrast: strongest window overlaps 0/9 base injections.
- Layer-18 User's strongest single window also overlaps 0/9 base injections, showing why unconditional top-one retrieval is insufficient. Its lower threshold produces multiple sustained candidates; the structural/directive filter and cap preserve the injected candidate.
- Full-distribution and maximum-non-Tool composite scores improve CoT-forgery localization but still have 0/9 base top-window hits.

Detailed metrics and the case-13 layer comparison are in [`ALL_ROLE_REANALYSIS.md`](ALL_ROLE_REANALYSIS.md).

## A genuinely role-general algorithm

### 1. Generate a feature vector for every role and layer

For each non-Tool role `r ∈ {System, User, CoT, Assistant}` and selected layer `l`, compute:

```text
probability:        p_l,r(t)
role margin:        z_l,r(t) - z_l,tool(t)
local contrast:     mean_w(p_l,r)(t) - mean_W(p_l,r)(t)
sustained fraction: fraction of a segment above a clean role quantile
segment features:   mean, q90, maximum, run length
```

Use several short windows, such as 16, 32, 64, and 128 tokens. A short or obfuscated injection need not form the same 64-token plateau as the present attacks.

### 2. Calibrate each role separately on clean Tool outputs

Let `F_clean(l,r,s,x)` be the clean distribution for layer `l`, role `r`, structural class `s`, and feature `x`. Structural classes should at least distinguish visible prose, code/quoted material, markup, URLs, and metadata.

Convert a segment's raw feature into a clean tail probability:

```text
q_l,r(segment) =
    P_clean(
        a clean segment of this structural type is at least this anomalous
    )

e_l,r(segment) = -log10(q_l,r(segment))
```

This is the key generalization. A System score is useful only when it is surprising for clean System-like content; it is not promoted merely because System is the largest logit.

### 3. Fuse roles and layers after calibration

A simple role-general candidate score is:

```text
E(segment) =
    max over layers l and non-Tool roles r of e_l,r(segment)
    + cross_layer_persistence_bonus
    + sustained_run_bonus
```

Record the winning `(layer, role)` as the candidate's explanation. Calibrate the maximum `E` at the **document level**, not only token-by-token, so scanning a 30k-token page is included in the false-positive estimate. Split-conformal calibration or an empirical maximum-score quantile can do this.

More conservative fusion can require agreement between adjacent layers. High-recall fusion can use a union over roles, but its final threshold must be calibrated on the union; otherwise adding roles silently multiplies false positives.

### 4. Form and rank semantic candidates

1. Turn sustained anomalous intervals into candidate spans.
2. Expand to the smallest meaningful sentence, paragraph, code block, or HTML block.
3. Merge overlapping signals from different roles/layers.
4. Retain the calibrated evidence for every contributing role.
5. Apply non-maximum suppression and return at most `K=3`.
6. Do not manufacture candidates when none meets the calibrated floor.

A generic directive-likeness score can help rank candidates, but it should not be the sole hard filter in a general detector because injections may be declarative, encoded, multilingual, or code-only.

### 5. Use tiered fallback behavior

A practical migration from the current selector is:

```text
Tier 1:
    current high-confidence CoT8 and User18 candidates

Tier 2:
    candidates from other roles whose clean-calibrated evidence is strong
    and persists across layers

Tier 3:
    if consequential tools are available and tiers 1–2 are empty,
    send at most one explicitly low-confidence semantic segment
```

Tier 2 or Tier 3 should not displace stronger Tier-1 candidates from the top-three budget. This lets new System/Assistant signals increase coverage without changing already successful CoT/User behavior.

### 6. Keep the LLM judge

Even a perfect role-confusion detector cannot decide whether a role-like instruction is authorized. Benign documents legitimately contain instructions, quotations, policies, code, and dialogue. The role probe should generate candidates; the judge should decide whether the candidate contains an instruction that lacks authority under the trusted user task.

## Key difficulty

The technical difficulty is not computing four separate plateaus. That is straightforward. The difficulty is determining whether each plateau is **unexpected under clean Tool content**, controlling the increased false-positive opportunity created by more roles, layers, windows, and page positions, and retaining enough candidate budget that the true span reaches the judge.

Adding four roles, three layers, and four window sizes creates 48 correlated scans before considering document position. A raw union almost always finds something unusual on a long page. Per-channel calibration followed by document-level calibration of the fused maximum is what turns that union into a statistically meaningful detector.

## How to describe the present experiment in a write-up

A defensible description is:

> We implement a role-indexed candidate detector and instantiate two channels selected on the development attack families: layer-8 CoT contrast for CoT-forgery and layer-18 User contrast for base injection. This is a proof of concept, not evidence that other role-confusion modes are unimportant. A general deployment should calibrate System, User, CoT, and Assistant channels separately on diverse clean Tool outputs and evaluate held-out attacks targeting every role.

It is reasonable to optimize the two current channels separately as an initial experiment, provided the limitation is explicit and the frozen evaluation is not used for further tuning.

## Required evaluation for the general version

The current corpus contains only base and CoT-forgery presentation families with one simulated exfiltration objective. Before claiming a general role-confusion detector, add held-out attack families deliberately styled as:

- System policies or administrator notices;
- User requests and quoted follow-up messages;
- CoT/reasoning continuations;
- Assistant answers or prior-turn continuations;
- mixtures and transitions between roles;
- subtle attacks that remain mostly Tool-like.

Report candidate recall by target role, recall at `K=1/3`, candidates per clean page, clean document candidate rate, judge false-positive rate, clean-task reversal, ASR, STSR, secure task success, and uncertainty. Freeze the calibration before the 200-case evaluation.
