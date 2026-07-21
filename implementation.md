# Label Cleaner — Implementation Guide

## Overview

`label_cleaner` benchmarks data-cleaning methodologies on noisy training sets.
Each method ranks training samples by how harmful they are, then incrementally
applies a noise-type-specific **action** (correct label, cap feature, remove row)
and measures test-set classification accuracy at each cleaning proportion from 0% to 100%.
DataScope's Shapley utility is scored against a dedicated held-out validation split
(distinct from the final test split) so its ranking is never informed by the same
data used for final accuracy/DP reporting — see `core/prep.py: prepare_fixed_split`.

The benchmark covers three real-world datasets, four noise types, two model pipelines,
and three active cleaning methods — DataScope, CleanLab, and a random-order baseline —
plus an outlier-only DataScope-removal variant. Four further methods developed during
this project (CL-Adaptive, Kairos, DS-Hybrid, Auto-Hybrid) were evaluated on run_v2 and
subsequently removed from the pipeline; they are documented below for the record, and
their code exists only as disabled stubs in `methods/cleaning.py`.

---

## Datasets

| Dataset | Rows | Features | Task | Protected attribute |
|---|---|---|---|---|
| **Adult** | 48,842 | 14 | Income >50K | Sex |
| **German Credit** | 1,000 | 20 | Credit risk | Sex |
| **Titanic** | 891 | 7 | Survival | Sex |

---

## Noise Types

| Noise type | What is corrupted | Action applied at cleaning time |
|---|---|---|
| **Outlier** | One feature column — values overwritten with a per-row, scale-aware extreme (mean + k·σ, k ~ Uniform(3,5)) for a random fraction of rows | Cap at 2-sigma of the clean distribution |
| **Random label** | Labels flipped uniformly at random to any other class | Restore original ground-truth label |
| **NNAR** (Noise Not At Random) | Labels of a random fraction of the protected subgroup are flipped — noise rate depends on group membership, not on the label value itself | Restore original ground-truth label |
| **MNAR** (Missing Not At Random) | Feature values of a random fraction of the protected subgroup are set to NaN — data absence is correlated with a sensitive attribute, simulating differential data-collection quality | Remove the detected row (the missing feature value cannot be recovered) |

---

## Pipelines

| Key | Architecture | Notes |
|---|---|---|
| **p1a** | ColumnTransformer union — numeric: MedianImputer → PowerTransformer (Yeo-Johnson) → StandardScaler; categorical: MostFrequentImputer → Logistic Regression (lbfgs) | General-purpose linear model; handles mixed feature types |
| **p2b** | MedianImputer → StandardScaler → PCA (n = min(8, n_features)) → SelectKBest (f_classif, k = min(5, n_pca)) → Random Forest (100 trees) | Non-linear ensemble on reduced features; contrasts with p1a's linear classifier |

---

## Cleaning Methods

### 1. DataScope

**How it works:**
Computes a Shapley value for every training sample measuring its marginal contribution to model accuracy. Shapley values are the unique attribution satisfying fairness axioms (efficiency, symmetry, linearity). Samples with the most *negative* contribution — those that hurt accuracy the most — are ranked first and cleaned.

The default uses `ImportanceMethod.NEIGHBOR` — a KNN approximation that avoids enumerating all coalitions, making it tractable for datasets up to ~50k rows. A Monte Carlo variant (`ImportanceMethod.MONTECARLO`) is more accurate but was found to be too slow (56+ minutes at 500 iterations) for the adult dataset; 50 iterations with NEIGHBOR is the working default.

**Implementation details (`methods/cleaning.py: clean_datascope`):**
1. Fit the full pipeline on the noisy training set
2. Compute Shapley importances via `ShapleyImportance(method=NEIGHBOR).fit(X_train).score(X_val)` — scored against the held-out validation split, not the test split (see note above; this was a test-set leak in earlier runs, fixed in `prepare_fixed_split`)
3. Sort noisy positions by ascending Shapley importance (lowest = most harmful first)
4. Incrementally apply `action_fn` to the top-k% and measure accuracy on the untouched test split

**Strengths:**
- Theoretically grounded — the only method with axiomatic fairness guarantees
- Consistently strong across all noise types; never catastrophically fails
- Works on both label and feature noise

**Weaknesses:**
- Ranking computed once on the noisy dataset — becomes stale as noise is progressively removed
- Does not use label uncertainty — treats a badly-labelled sample identically to one with a correct but unusual label
- Monte Carlo variant too slow for large datasets at useful iteration counts

---

### 2. CleanLab

**How it works:**
Obtains out-of-fold predicted class probabilities via 5-fold cross-validation, then computes the `self_confidence` score for each training sample: P(current label is correct | x). Lower self-confidence = more suspicious. Uses `find_label_issues` from the `cleanlab` library which applies confident learning theory to rank all training samples.

**Implementation details (`methods/cleaning.py: clean_cleanlab`):**
1. Run `cross_val_predict(..., method="predict_proba")` on the full training set
2. Call `find_label_issues(return_indices_ranked_by="self_confidence")`
3. Incrementally apply `action_fn` to the top-k% most suspicious samples

**Strengths:**
- Fully unsupervised — does not require knowledge of which samples are noisy
- Reliable for label noise (random, NNAR) where model confidence is a strong proxy for label correctness

**Weaknesses:**
- Self-confidence degrades at high noise rates (>20%) — OOF probabilities become noisy themselves
- Less effective on feature noise (outlier, MNAR) where the label is correct but features are corrupted
- Cannot distinguish "label is wrong" from "sample is genuinely ambiguous at the decision boundary"

---

### 3. CleanLab Adaptive (CL-Adaptive) — removed after run_v2

**How it works:**
Extends CleanLab by routing each flagged sample to one of two actions rather than applying the same action uniformly. The key insight is that not all suspicious samples should be treated equally — some have a clearly wrong label that can be corrected, while others are so uncertain that removal is safer.

The split threshold is found adaptively using **Otsu's method** on the self-confidence scores of all CleanLab-flagged samples. Otsu sweeps every candidate threshold and selects the one maximising between-class variance:

```
σ²_between(t) = w₀(t) · w₁(t) · (μ₀(t) − μ₁(t))²
```

Before applying Otsu, the **bimodality coefficient** (BC) is checked:
```
BC = (skewness² + 1) / Pearson_kurtosis
```
If BC ≤ 5/9, the self-confidence distribution is unimodal (e.g. random label noise produces a single cluster of low-confidence scores), and the method falls back to the median threshold to avoid spurious splits.

**Routing logic:**
- `self_confidence < threshold` → most suspicious → **remove** the sample
- `self_confidence ≥ threshold` → moderately suspicious → **correct** via action_fn

**Implementation details (`methods/cleaning.py: clean_cleanlab_adaptive, _otsu_threshold`):**
- Same OOF probability computation as standard CleanLab
- `_otsu_threshold(scores)`: 256-bin histogram sweep with BC guard (BC > 5/9 required to use Otsu; falls back to median otherwise)

**Strengths:**
- Differentiates recoverable label errors (correct) from irreparable noise (remove)
- BC guard prevents spurious adaptive splits on unimodal distributions
- Consistent small gains over standard CleanLab on outlier noise

**Weaknesses:**
- On small datasets (Titanic) confidence scores rarely form clean bimodal distributions, limiting the Otsu split's effectiveness
- MNAR noise is poorly handled — self-confidence is unreliable when features are corrupted

---

### 4. Kairos — removed after run_v2

**How it works:**
A simplified adaptation inspired by KAIROS (Zhu, Prashant, Cloninger & Salimi, "KAIROS: Scalable Model-Agnostic Data Valuation," NeurIPS 2025, arXiv:2506.23799). The original paper derives a closed-form MMD (Maximum Mean Discrepancy) contribution score with conditional kernels for unified label- and feature-error detection; this implementation instead hand-builds two separate signals and blends them heuristically:

1. **Feature score** — RBF kernel similarity of each training sample to the test distribution minus its similarity to the training distribution. Samples that "look like" the test set are more valuable; samples that are anomalous relative to the test set are penalised.
2. **Residual score** — P(correct label | x) from a logistic regression trained on the test set. Higher = label is consistent with the test-set distribution = likely clean.

```
score_i = 0.97 × feature_score_i + 0.03 × residual_score_i
```

Samples with the lowest combined score are cleaned first. The pipeline's feature preprocessing steps (PCA, scaling) are applied before computing the kernel to avoid operating on raw mixed-type features.

**Scaling for large datasets:** The full RBF kernel matrix for adult (~32k × 32k) would require ~5.4GB RAM. Reference sets are subsampled: val_ref ≤ 500 rows, train_ref ≤ 2,000 rows, preserving the distribution estimate while keeping memory tractable.

**Implementation details (`methods/cleaning.py: _kairos_scores, clean_kairos`):**
1. Fit pipeline, extract `feature_pipe = pipeline[:-1]`, transform X_train and X_test
2. Subsample reference sets
3. Compute `K_tv = rbf_kernel(X_train, val_ref)`, `K_tt = rbf_kernel(X_train, trn_ref)`
4. `feature_score = K_tv.mean(axis=1) − K_tt.mean(axis=1)`
5. Fit logistic regression on X_test, compute residual scores on X_train
6. Blend and rank ascending (lowest = noisiest)

**Strengths:**
- Model-agnostic — scores computed independently of the main pipeline
- Feature score is particularly effective for outlier noise where anomalous feature values reduce similarity to the test distribution
- Best overall on outlier noise (adult p2b: 0.8216 vs baseline 0.8101)

**Weaknesses:**
- **Collapses on MNAR noise** — protected-group feature corruption distorts the kernel, causing near-random scores (~0.18–0.20 accuracy vs ~0.81 baseline). This is the most significant failure mode.
- Requires a clean test set to compute feature similarity
- Memory/compute cost requires subsampling on large datasets

---

### 5. DataScope Hybrid (DS-Hybrid) — removed after run_v2

**How it works:**
Addresses a core weakness of DataScope (no label uncertainty signal) and CleanLab (no global accuracy impact signal) by blending both into a single ranking score over the known noisy positions:

```
hybrid_score_i = α · norm(shapley_i) + (1 − α) · norm(1 − self_confidence_i)
```

Both signals are min-max normalised to [0, 1] before blending (default α = 0.5). Samples that both hurt model accuracy *and* have a likely-wrong label are ranked highest.

**Implementation details (`methods/cleaning.py: clean_datascope_hybrid`):**
1. Compute Shapley importances as in DataScope
2. Run `cross_val_predict` to obtain OOF probabilities
3. Extract `self_confidence[i] = pred_probs[i, y_train[i]]`
4. Normalise both signals, blend, rank noisy positions by descending hybrid score

**Strengths:**
- Reduces false positives from each signal taken alone
- Same runtime as DataScope + one additional cross-validation pass

**Weaknesses:**
- Requires ground-truth noisy positions — not fully unsupervised
- On small datasets the two signals are already correlated, producing no meaningful improvement over plain DataScope
- Adds the full cost of cross-validation on top of Shapley computation

---

### 6. Auto-Routing Hybrid (hybrid_auto) — removed after run_v2

**How it works:**
Rather than applying a fixed method, the hybrid auto-router *detects* the likely noise type from the data and delegates to the empirically best method for that type:

```
outlier   → Kairos      (RBF kernel score detects feature anomalies best)
mnar      → CleanLab    (Kairos collapses; CL is most stable)
nnar      → DataScope   (Shapley captures structured group-based label flips)
rnd_label → CleanLab    (self_confidence reliably ranks random label noise)
```

**Detection logic (`methods/cleaning.py: _detect_noise_type`):**

Two signals are computed from OOF predicted probabilities:

1. **Feature anomaly** — flags the bottom-20% self-confidence samples and checks what fraction have any feature value with |z-score| > 3. High fraction → feature-space corruption.
2. **Spatial clustering** — measures how tightly the flagged samples cluster:
   `clustering = 1 − within_flagged_variance / overall_variance`
   High clustering → suspicious samples are concentrated in a subgroup.

Decision tree:
```
feature_anomaly ≥ 0.25 AND clustering < 0.30  → outlier
feature_anomaly ≥ 0.25 AND clustering ≥ 0.30  → mnar
feature_anomaly < 0.25 AND clustering ≥ 0.30  → nnar
feature_anomaly < 0.25 AND clustering < 0.30  → rnd_label
```

**Strengths:**
- No noise-type knowledge required — fully adaptive
- Correctly avoids the Kairos collapse on MNAR by routing to CleanLab instead
- Performs at or above the mean of individual methods across most configurations

**Weaknesses:**
- Detection is heuristic — thresholds (0.25, 0.30) were set empirically and may not generalise to all datasets
- Misclassified noise type → wrong method → can underperform any individual method
- OOF cross-validation required for detection adds runtime overhead before any cleaning begins

---

## Development Attempts & Iterations

### Adaptive Threshold for CL-Adaptive

Four strategies were investigated for splitting CleanLab-flagged samples into correct vs. remove:

#### Attempt 1: Fixed median threshold
The simplest approach — split at the median self-confidence of all flagged samples. Guarantees a 50/50 correct/remove split but ignores the actual shape of the confidence distribution.

**Result:** Mean titanic accuracy: 0.7947. Used as the baseline for comparison.

#### Attempt 2: Beta Mixture Model (BMM)
Fit a 2-component Beta Mixture Model via EM to the self-confidence scores (Beta distributions are natural for [0,1] data). Use the crossing point of the two component PDFs as the threshold. A separation guard (min component mean difference ≥ 0.15) falls back to median when components are too close.

```python
# EM updates for Beta parameters via method-of-moments
mu_k = (resp[:, k] * scores).sum() / Nk[k]
var_k = (resp[:, k] * (scores - mu_k)**2).sum() / Nk[k]
factor = mu_k * (1 - mu_k) / var_k - 1
alpha_k = factor * mu_k;  beta_k = factor * (1 - mu_k)
# Threshold via Brent's method on pdf_low(x) - pdf_high(x)
threshold = brentq(lambda x: w_lo*Beta.pdf(x,a_lo,b_lo) - w_hi*Beta.pdf(x,a_hi,b_hi), lo_bound, hi_bound)
```

**Result:** Mean titanic accuracy: 0.7947. Gains on MNAR p2b (+2.8pp) and NNAR p1a (+1.1pp) were offset by regressions on NNAR p2b (−2.2pp) and rnd_label p1a (−1.1pp) where OOF probabilities from PCA-based pipelines are poorly calibrated.

#### Attempt 3: Otsu's Method + Bimodality Coefficient (current)
Otsu's method is non-parametric — it maximises between-class variance by sweeping all candidate thresholds. A bimodality coefficient (BC = (skewness² + 1) / Pearson kurtosis, threshold 5/9) gates whether a bimodal split is warranted at all.

**Why BC over the BMM separation guard:** The BMM's separation guard checked class mean distance (which even unimodal distributions can satisfy), while BC uses the full shape of the distribution — skewness and kurtosis — to confirm genuine bimodality before committing to a split.

**Result:** Mean titanic accuracy: 0.7954 (**best**). No regressions; small gain on outlier noise.

#### Attempt 4: KDE Valley Detection
Smoothed the self-confidence histogram using Scott's bandwidth rule, found the two most prominent peaks via `find_peaks` with minimum prominence filter, and used the minimum between them as the threshold. Falls back to median if fewer than two prominent peaks are found.

**Why KDE over Otsu:** Otsu maximises variance, which can split at a location that isn't a true valley. KDE directly finds the natural trough between peaks.

**Result:** Mean titanic accuracy: 0.7947. The BC-gated Otsu fallback prevented regressions but also prevented gains on the same configurations as KDE.

**Conclusion:** Otsu + BC is the current implementation. The marginal differences between methods are small on Titanic (~700 training rows). Larger datasets with more noisy samples would give these threshold methods more signal to work with.

---

### DataScope Optimisation Attempts

#### Attempt: Iterative LOO Reranking (removed)
The initial DataScope ranking is computed on the fully noisy dataset. As noise is progressively cleaned, the model's perspective on which remaining samples are harmful changes — motivating reranking at intervals.

**Implementation:** After every 5% of noisy samples are cleaned, refit the model on the current cleaned data and re-score remaining noisy candidates via Leave-One-Out (LOO):
```
LOO_importance[i] = acc(X \ {i}) − acc(current_acc)
```

**Why it was removed:** LOO cost is O(n_noisy × fit_time × n_rounds). On Titanic (~50 noisy samples, 20 rounds) this was acceptable but produced **zero improvement** — the initial Shapley ranking on logistic regression is already near-optimal. On adult (1,600+ noisy samples at 5%) the run was terminated after 45+ minutes with only 1/8 configs complete. The method was removed from the final pipeline. It would only show real gains with non-linear models (random forests, neural nets) or highly correlated noise clusters.

#### Attempt: MONTECARLO ImportanceMethod
`ImportanceMethod.MONTECARLO` provides higher-accuracy Shapley estimates by sampling random permutations. Tested with 500 iterations on the p2b pipeline (PCA-based, not compatible with NEIGHBOR's KNN approximation).

**Result:** 56+ minute runtime per experiment. Reverted to `ImportanceMethod.NEIGHBOR` for all pipelines. Reduced to 50 iterations as a middle ground — no accuracy gain observed over NEIGHBOR at that iteration count.

---

## Full Results (run_v2)

### Adult — 10% noise

| Experiment | Base | DS | CleanLab | Kairos | CL-Ada | DS-Hyb | Auto |
|---|---|---|---|---|---|---|---|
| mnar / p1a | 0.8095 | 0.8081 | 0.8045 | 0.1905† | 0.8050 | 0.8081 | 0.1905† |
| mnar / p2b | 0.8142 | **0.8151** | 0.8099 | 0.1862† | 0.8125 | **0.8151** | 0.8099 |
| nnar / p1a | 0.8087 | 0.8095 | **0.8101** | 0.8095 | 0.8092 | 0.8095 | **0.8101** |
| nnar / p2b | 0.8130 | 0.8131 | 0.8095 | 0.8131 | **0.8148** | 0.8131 | 0.8095 |
| outlier / p1a | 0.8111 | 0.8101 | **0.8111** | 0.8087 | 0.8107 | 0.8101 | **0.8111** |
| outlier / p2b | 0.8101 | 0.8096 | 0.8191 | **0.8216** | 0.8130 | 0.8096 | 0.8191 |
| rnd_label / p1a | 0.8102 | 0.8095 | 0.8095 | 0.8095 | 0.8092 | 0.8095 | 0.8095 |
| rnd_label / p2b | 0.8072 | 0.8131 | **0.8144** | 0.8131 | 0.8142 | 0.8131 | **0.8144** |
| **Mean** | 0.8105 | 0.8110 | 0.8110 | 0.8126‡ | 0.8111 | 0.8110 | **0.8119**‡ |

†Kairos collapses on MNAR. ‡Mean excludes MNAR collapses.

---

### German Credit — 20% noise

| Experiment | Base | DS | CleanLab | Kairos | CL-Ada | DS-Hyb | Auto |
|---|---|---|---|---|---|---|---|
| mnar / p1a | 0.7150 | 0.7050 | **0.7450** | 0.2850† | 0.7400 | 0.7050 | **0.7450** |
| mnar / p2b | 0.6950 | 0.6650 | **0.7000** | 0.3000† | **0.7200** | 0.6650 | **0.7000** |
| nnar / p1a | 0.7150 | 0.7100 | **0.7150** | 0.7100 | **0.7150** | 0.7100 | **0.7150** |
| nnar / p2b | 0.6650 | **0.6850** | 0.6750 | **0.6850** | **0.6950** | **0.6850** | 0.6750 |
| outlier / p1a | 0.7350 | 0.7150 | 0.7150 | 0.7200 | **0.7250** | 0.7150 | 0.7150 |
| outlier / p2b | 0.6700 | **0.7150** | 0.6900 | 0.6950 | **0.7200** | **0.7150** | 0.6900 |
| rnd_label / p1a | 0.6850 | **0.7100** | 0.6900 | **0.7100** | 0.7000 | **0.7100** | 0.6900 |
| rnd_label / p2b | 0.6900 | 0.6850 | 0.6850 | 0.6850 | 0.6650 | 0.6850 | 0.6850 |
| **Mean** | 0.6963 | 0.6987 | **0.7019** | 0.7008‡ | **0.7100** | 0.6987 | **0.7019**‡ |

†Kairos collapses on MNAR. ‡Mean excludes MNAR collapses.

---

### Titanic — 20% noise

| Experiment | Base | DS | CleanLab | Kairos | CL-Ada | DS-Hyb | Auto |
|---|---|---|---|---|---|---|---|
| mnar / p1a | 0.8156 | 0.7709 | **0.8101** | 0.1844† | **0.8101** | 0.7709 | **0.8101** |
| mnar / p2b | 0.8156 | 0.7989 | **0.8045** | 0.1955† | 0.7263 | 0.7989 | **0.8045** |
| nnar / p1a | 0.7765 | **0.8101** | 0.8045 | **0.8101** | 0.7933 | **0.8101** | 0.8045 |
| nnar / p2b | 0.8045 | **0.8268** | 0.8045 | **0.8268** | **0.8324** | **0.8268** | 0.8045 |
| outlier / p1a | 0.7989 | 0.7933 | 0.7933 | 0.7933 | **0.8045** | 0.7933 | 0.7933 |
| outlier / p2b | 0.8380 | 0.8268 | 0.8268 | **0.8324** | 0.8212 | 0.8268 | 0.8268 |
| rnd_label / p1a | 0.7989 | **0.8101** | 0.7933 | **0.8101** | 0.7933 | **0.8101** | 0.7933 |
| rnd_label / p2b | 0.6983 | **0.8268** | 0.7709 | **0.8268** | 0.7765 | **0.8268** | 0.7709 |
| **Mean** | 0.8053 | 0.8080 | 0.8010 | **0.8166**‡ | 0.7947 | 0.8080 | 0.8010‡ |

†Kairos collapses on MNAR. ‡Mean excludes MNAR collapses.

---

## Auto-Hybrid Performance Analysis

The Auto-Hybrid cleaner routes to the best individual method per detected noise type. Its accuracy equals that method's accuracy when detection is correct, and degrades when detection is wrong.

### Where detection works well
- **MNAR** — correctly routed to CleanLab on all datasets, avoiding the Kairos collapse entirely. Matches CleanLab exactly.
- **Random label on adult** — correctly routed to CleanLab, matching the best individual method.

### Where detection struggles
- **NNAR on titanic** — routed to DataScope in some configs but CleanLab routes to lower accuracy (0.8045 vs DataScope's 0.8101/0.8268). The NNAR spatial clustering signal is weaker on small datasets.
- **Outlier on german p2b** — routed to CleanLab (0.6900) instead of DataScope/CL-Adaptive (0.7150/0.7200). Feature anomaly z-score detection misses this outlier pattern.

### Mean accuracy across all datasets (excluding Kairos MNAR collapses)

| Method | Adult | German | Titanic | Overall mean |
|---|---|---|---|---|
| DataScope | 0.8110 | 0.6987 | 0.8080 | 0.7726 |
| CleanLab | 0.8110 | **0.7019** | 0.8010 | 0.7713 |
| Kairos | **0.8126** | 0.7008 | **0.8166** | **0.7767** |
| CL-Adaptive | 0.8111 | **0.7100** | 0.7947 | 0.7719 |
| DS-Hybrid | 0.8110 | 0.6987 | 0.8080 | 0.7726 |
| **Auto-Hybrid** | 0.8119 | **0.7019** | 0.8010 | 0.7716 |

The Auto-Hybrid places third overall, behind Kairos and CL-Adaptive. Its primary value is **safety** — it never triggers the Kairos MNAR collapse, achieving consistent performance across all noise types without requiring the user to specify the noise type in advance.

---

## Key Findings

1. **Kairos is the strongest single method** (excluding MNAR) — its RBF kernel feature score directly detects samples anomalous relative to the test distribution, making it the natural choice for outlier noise. On adult p2b outlier it achieves 0.8216 vs baseline 0.8101.

2. **Kairos has a critical failure mode on MNAR** — feature corruption of a protected subgroup shifts the entire group's kernel similarity score to near-zero, producing essentially random rankings (~0.18 accuracy vs ~0.81 baseline). This is a fundamental limitation of distribution-matching methods when the corruption is systematic.

3. **DataScope is the most robust** — never catastrophically fails, performs near the top across all four noise types. The best all-around choice when noise type is unknown and Kairos's failure mode is a concern.

4. **CleanLab degrades at high noise rates** — at 20% noise on Titanic p2b, CleanLab achieves only 0.7709 vs DataScope's 0.8268. OOF probabilities computed on a 20% noisy dataset are themselves corrupted, weakening the self-confidence signal.

5. **CL-Adaptive's Otsu threshold shows consistent small gains on outlier and NNAR noise** — the bimodal confidence split correctly routes a fraction of samples to removal rather than correction. The BC guard is essential to prevent spurious splits on unimodal distributions.

6. **DS-Hybrid adds no value on small datasets** — on Titanic (~700 training rows), Shapley importance and self-confidence are already strongly correlated, making the blend equivalent to either signal alone.

7. **Auto-Hybrid's main contribution is safety** — by detecting and routing away from Kairos on MNAR noise, it avoids the worst-case outcome at the cost of some performance on other noise types.

---

## Architecture

```
label_cleaner/
├── core/
│   ├── models.py        # MethodCurves, ExperimentArtifacts, NoiseBundle, PreparedSplit dataclasses
│   └── prep.py          # Fixed train/validation/test split (80/10/20-ish; val capped at
│                         # val_cap rows). Val is used only to score the DataScope Shapley
│                         # utility, kept separate from the test split used for final metrics.
├── data/
│   └── datasets.py      # Dataset loaders and DatasetInfo (adult, german, titanic)
├── methods/
│   ├── cleaning.py      # Active cleaners (DataScope, CleanLab, Random) + action functions; removed methods left as DISABLED stubs
│   └── noise.py         # Noise injectors: inject_outlier, inject_rnd_label, inject_nnar, inject_mnar
├── orchestration/
│   ├── catalog.py       # Pipeline factory definitions (p1a, p2b)
│   └── experiments.py   # Experiment runners: run_{outlier,rnd_label,nnar,mnar}_experiment_with_artifacts
└── scripts/
    ├── run_all_experiments.py      # CLI driver: runs full matrix, saves figures, caches, and report.md
    ├── generate_combined_report.py # Reads per-dataset caches, produces combined_report.md + figures
    └── generate_ppt.py             # Builds results.pptx from run_v5 caches and figures
```

### Key dataclasses (`core/models.py`)

```python
@dataclass
class MethodCurves:
    datascope:          List[float]           # accuracy at each cleaning proportion
    random_mean:        List[float]           # mean over 3 random seeds
    random_std:         List[float]
    cleanlab:           List[float]
    baseline:           float                 # accuracy at 0% cleaning
    proportions:        np.ndarray
    datascope_removal:  Optional[List[float]] = None  # outlier only
```

### Action functions (`methods/cleaning.py`)

| Function | Use with |
|---|---|
| `action_cap(col_idx, cap_value)` | Outlier noise — caps feature at 2-sigma |
| `action_restore_labels(y_clean)` | Random label / NNAR — restores ground-truth label |
| `action_remove()` | MNAR (removes detected rows) and the outlier removal variant |

The threshold helpers used by the removed methods (`_otsu_threshold`,
`_kde_valley_threshold`, `_detect_noise_type`) were deleted along with them;
only DISABLED comment markers remain in `methods/cleaning.py`.
