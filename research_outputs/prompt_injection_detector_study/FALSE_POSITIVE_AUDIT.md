# False-positive audit of the role-probe detector

## Direct answer

The current role probe gives a useful **ranking/localization signal**, but it does **not** yet give a standalone detector with both few false positives and few false negatives.

This conclusion uses every available ordinary token, not only the highlighted injection regions:

- 20 clean pages for threshold calibration;
- 15 separate page-bearing clean controls for false-positive measurement;
- 18/20 attack pages whose Tool results and injection spans reached the model;
- 5 baseline-successful attacks among those 18.

The other two attack cases returned only a short error Tool result and therefore had no injected text for a detector to observe. All numbers are development-set diagnostics, not frozen 200-case test results.

## Practical operating points

| Detector and threshold | Attack detection | Baseline-success detection | Held-out clean pages falsely flagged |
| --- | ---: | ---: | ---: |
| Layer 8, `p(user)+p(CoT)` local contrast 64 vs 512, 5% clean-calibration budget | 11/18 (61.1%) | 4/5 (80.0%) | 0/15 (0.0%) |
| Same detector, 20% calibration budget | 11/18 (61.1%) | 4/5 (80.0%) | 3/15 (20.0%) |
| Layer 18, robust user/CoT margin mean-64, 5–20% calibration budgets | 8/18 (44.4%) | 3/5 (60.0%) | 1/15 (6.7%) |
| Layer 18, `p(user)+p(CoT)` contrast 64 vs 512, 10% calibration budget | 14/18 (77.8%) | 5/5 (100.0%) | 5/15 (33.3%) |

Thus there is a usable low-false-positive mode, but it misses 7/18 known injections. Detecting all five attacks that actually succeeded is possible in this sample, but the tested operating point falsely flags one third of clean pages.

## Cost of demanding very high recall

Thresholds below were chosen retrospectively to reach the requested attack recall, so they are optimistic rather than deployable estimates. Among the tested layer-18 role-only scores:

| Required attack recall | Actual detected | Best observed held-out clean FPR |
| --- | ---: | ---: |
| At least 80% | 15/18 (83.3%) | 7/15 (46.7%) |
| At least 90% | 17/18 (94.4%) | 7/15 (46.7%) |
| 100% | 18/18 (100.0%) | 8/15 (53.3%) |

Layer 8 was worse in the high-recall regime: reaching 100% recall falsely flagged at least 12/15 held-out clean pages among the tested scores.

This is the clearest answer to the false-positive concern: **with the current probe alone, low false negatives require an unacceptably high clean-page false-positive rate.**

## Why the injection plots looked good

Token-level separation and document-level detection are different problems. A smoothed layer-18 `p(user)+p(CoT)` trace reaches token AUROC 0.931 with a 128-token window. That means an injection token usually ranks above a randomly selected ordinary token.

But each page contains roughly 20,000–40,000 tokens. A document detector scans thousands of overlapping windows and normally uses the maximum. Even a rare benign spike will eventually occur in a long page. This multiple-scanning/extreme-value effect explains why a good token ROC can coexist with poor clean-document false-positive rates.

## What the strongest non-injection peaks contain

The maximum-scoring ordinary windows were usually not natural-language instructions. They were Wikipedia/MediaWiki markup and boilerplate such as:

| Source | Representative highest-scoring benign text |
| --- | --- |
| Navigation markup | `cdx-button--fake-button ... vector-icon-appearance ... vector-dropdown-label-text` |
| Table-of-contents markup | `data-feature-name="toc-pinned" ... data-pinned-container-id ...` |
| URL-encoded text | `%D0%A0%D0%95%D0%9F...` |
| Citation metadata | `ctx_ver=Z39.88-2004&rft_val_fmt=...` |
| Wiki templates/lists | `[[American football|Football]] ... [[Rifle]] ...` |
| Taxonomic HTML | `<i>Necrophorus</i> ... <small>Leach & Nodder ...` |

Some of the same benign peaks occur at exactly corresponding positions in paired clean and injected versions of a page. This is evidence that the probe is reacting to markup/style rather than an injected instruction in those regions.

## What should be changed

The best near-term design is a cascade:

1. Parse HTML and scan visible semantic text blocks separately from markup, scripts, URLs, citation metadata, and navigation boilerplate. The observed top false positives make this the highest-priority change.
2. Use the probe only to propose suspicious spans. Aggregate user/tool and CoT/tool margins over sentences or 16/32/64-token windows and require persistence.
3. Verify each proposed span with an independent instruction/conflict detector: does the span direct the agent to take an action, and is that action unrelated to or in conflict with the trusted user task?
4. Add cross-layer agreement and, if available, attention transfer from the trusted user instruction to the suspicious Tool span.
5. Calibrate the maximum final span score at the **whole-document** level on clean Tool results of comparable length. Token-level calibration is insufficient.
6. For high recall, use a low threshold only as a review/quarantine trigger. Tool calls with external side effects or sensitive-file access should be blocked independently even if the detector is uncertain.

The role probe should therefore be treated as a high-value feature, not as the entire detector.

## Where to inspect

- `token_roc.svg` evaluates injection tokens against all ordinary page tokens.
- `token_heatmaps.html` shows the injection neighborhood and exposes every role logit on hover.
- `token_logit_excerpts.csv` contains the corresponding exact numeric values.
- `REPORT.md` contains the broader detector proposal and research references.
- `detector_metrics.json` contains the machine-readable operating points.
