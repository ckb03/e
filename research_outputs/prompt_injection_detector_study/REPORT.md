# Detecting prompt injection from the current role probe

## Answer in brief

Yes, useful token-level data remain. Raw five-role logits were already stored for layer 18, and this study exactly reconstructed layer-8 logits as `W h + b` from saved pre-MLP activations. This covers the 20-case attack-development suite and clean controls, **not the frozen 200-case final evaluation**. Two attack cases never returned a page, so injection detection is measurable on 18/20 attack cases.

The probe signal can localize injected text better than chance, but a single-token or single-threshold detector is not reliable enough. The existing layer-8 Mahalanobis/local score has injection-vs-page token AUROC **0.868**, yet its old threshold fires on **29.6%** of ordinary tokens in attacked pages. The raw-logit experiments below reach high token AUROC after smoothing, while clean-document false positives remain the limiting issue.

## Saved and generated artifacts

- `token_logits/layer-08/`: exact reconstructed logits for every page-bearing saved Tool result.
- `token_logit_excerpts.csv`: all five logits and probabilities for each injection plus 64 surrounding tokens.
- `token_heatmaps.html`: hoverable word/token view; gold underline is ground-truth injection.
- `token_roc.svg`: token-level localization ROC curves.
- `case-008-trace.svg`, `case-013-trace.svg`, `case-017-trace.svg`, and `case-020-trace.svg`: representative role-probability traces.
- `detector_metrics.json`: machine-readable results and operating points.

## What the logits look like

![Token ROC](token_roc.svg)

Smoothing matters because injected requests are contiguous spans, whereas ordinary HTML produces isolated role-logit spikes. The improvement from token score to 64–128-token averages supports a segment detector rather than a token threshold. However, maximizing a score anywhere in a 20k–40k-token page creates an extreme-value problem: even benign pages eventually contain a high-scoring region.

Representative traces (yellow is the exact injection):

![Case 8](case-008-trace.svg)

![Case 13](case-013-trace.svg)

![Case 17](case-017-trace.svg)

![Case 20](case-020-trace.svg)

For word-level inspection, open [the interactive heatmap](token_heatmaps.html) and hover tokens to see all logits.

## Layer-8 detector comparison

The final four columns use a threshold permitting at most a 20% false-positive budget on 20 clean calibration pages. “Held-out clean” contains the available clean-sanity and paired clean pages. These are development diagnostics, not final-test estimates.

| Score | Token AUROC | Macro AUROC | Top-128 overlaps injection | Attack recall | Baseline-success recall | Held-out clean FPR |
| --- | --- | --- | --- | --- | --- | --- |
| robust-UC mean-64 | 0.908 | 0.909 | 33.3% | 0.0% | 0.0% | 40.0% |
| robust-UC mean-32 | 0.894 | 0.893 | 33.3% | 0.0% | 0.0% | 40.0% |
| UC-margin mean-64 | 0.858 | 0.847 | 0.0% | 16.7% | 20.0% | 13.3% |
| pUC contrast 128-1024 | 0.783 | 0.786 | 50.0% | 66.7% | 80.0% | 40.0% |
| pUC mean-128 | 0.760 | 0.687 | 22.2% | 38.9% | 60.0% | 20.0% |
| pUC mean-64 | 0.740 | 0.672 | 27.8% | 33.3% | 60.0% | 26.7% |
| pUC mean-32 | 0.730 | 0.662 | 27.8% | 50.0% | 60.0% | 26.7% |
| pUC token | 0.709 | 0.649 | 22.2% | 38.9% | 80.0% | 26.7% |
| all-role-margin mean-64 | 0.674 | 0.739 | 33.3% | 0.0% | 0.0% | 33.3% |
| pUC contrast 64-512 | 0.672 | 0.702 | 50.0% | 61.1% | 80.0% | 20.0% |

The best development configuration by attack recall at this operating point is **pUC contrast 128-1024**. Its threshold tradeoff is:

| Clean calibration budget | Threshold | Calibration FPR | Held-out clean FPR | All-attack recall | Base recall | CoT recall | Recall of baseline successes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5.0% | 0.4138 | 5.0% | 13.3% | 55.6% | 11.1% | 100.0% | 80.0% |
| 10.0% | 0.4035 | 10.0% | 20.0% | 55.6% | 11.1% | 100.0% | 80.0% |
| 20.0% | 0.3879 | 20.0% | 40.0% | 66.7% | 33.3% | 100.0% | 80.0% |

Small denominators matter: attack recall is over 18 observable attacks and held-out clean FPR over 15 pages. A single attack changes recall by 5.6 percentage points.

## Recommended algorithm: segmented multi-signal conformal detector

Do not deploy `max token Userness/CoTness > tau`. Use this pipeline:

1. **Segment the Tool result.** Parse markup where possible and split visible content into semantic blocks/sentences, preserving a token map. PromptLocate independently argues for semantic segmentation before instruction localization ([paper](https://arxiv.org/abs/2510.12252)).
2. **Compute role-confusion features at layers 8, 16, and 18.** For token `t` and layer `l`, retain the full vector `z_l,t`; use margins `z_user-z_tool`, `z_cot-z_tool`, `logsumexp(z_user,z_cot)-z_tool`, entropy, and cross-layer persistence. The role-confusion work specifically finds that style can override the architectural role and that Userness/CoTness tracks attacks ([project and paper](https://role-confusion.github.io/)).
3. **Aggregate within segments, not isolated tokens.** Use mean, 90th percentile, fraction above a benign quantile, maximum 16/32/64-token moving average, and a local-background contrast. Require a sustained run or high segment score.
4. **Add two signals the role probe does not contain.** (a) policy-conflict alignment learned from safe/unsafe contrastive pairs, following the motivation of PVDetector ([paper](https://arxiv.org/abs/2607.12624)); and (b) attention transfer from the trusted task to the candidate segment, motivated by Attention Tracker ([paper](https://aclanthology.org/2025.findings-naacl.123/)). RAP-ID likewise fuses directive likeness, attention takeover, and policy conflict instead of trusting one internal signal ([paper](https://aclanthology.org/2026.findings-acl.738/)).
5. **Train a small segment classifier with grouped splits.** Inputs should be only the aggregate internal features plus structural metadata, not attack-template keywords. Split by underlying page and attack template; hold out entire attack families. Optimize a high-recall objective or cost-sensitive loss.
6. **Calibrate at the document level.** Define the document score as the maximum segment score, then set the threshold from maximum scores on clean Tool documents. Use split-conformal p-values so document length and multiple scanning are included in calibration. Maintain two thresholds: quarantine/high-confidence and review/low-confidence.
7. **Fail safely before tool execution.** When flagged, prevent sensitive or external side-effect tool calls, but allow the model to continue the requested summarization on sanitized content. Detection alone should not be the sole security boundary.

Suggested score for the role-only first stage:

```text
r_l,t = logsumexp(z_l,t,user, z_l,t,cot) - z_l,t,tool
s(segment) = classifier([
  mean(r), q90(r), max_mean_16(r), max_mean_32(r), max_mean_64(r),
  fraction(r > benign_q99), local_background_contrast(r),
  same features at layers 8/16/18, cross_layer_agreement
])
document_score = max_segment s(segment)
```

For low false negatives, choose the review threshold to target ≥99% recall on a much larger attack-development corpus, then use the second-stage policy/attention signals to reduce false positives. The present 18 observable attacks are far too few to establish 99% recall.

## Why an ensemble is necessary

- Role logits capture **source/role impersonation**, not whether an instruction is harmful or conflicts with the task.
- Natural prose, quoted dialogue, documentation, and HTML/JavaScript can look user-like or reasoning-like, creating false positives.
- A semantically subtle injection can remain Tool-like and evade the role probe, creating false negatives.
- Activation detectors are adaptive attack surfaces: work on evasive injections demonstrates that linear activation probes can be deliberately bypassed ([paper](https://arxiv.org/abs/2602.00750)). Randomized or multi-layer ensembles and adversarial detector training are therefore important.
- Stronger architectural/training defenses remain complementary: structured-query training separates instructions from data ([StruQ](https://arxiv.org/abs/2402.06363)), while instruction-hierarchy training teaches models to ignore lower-privilege conflicts ([paper](https://arxiv.org/abs/2404.13208)).

## Limitations

- This is the fixed 20-case attack-development set with only two attack families and 18 observable Tool results.
- Ground-truth injection boundaries were used only for offline evaluation and plots, never as detector input.
- Candidate selection and evaluation share the same small development set, so the reported best method is optimistic.
- The logits precede the evaluated steering interventions; steered-run token logits were not logged.
- No claim is made about the frozen 200-case test set or adaptive attacks until logits are collected and thresholds are frozen before evaluation.
