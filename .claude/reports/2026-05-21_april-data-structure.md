# April dataset — structure, main issues, recoverable signal

*2026-05-21 · companion to `notebooks/april/01b_single_network_walkthrough.ipynb`
and `notebooks/april/01c_single_patient_walkthrough.ipynb`*

---

## 1. What is in the batch

`data/correlation_mat_april_data/` — 226 pickles, all $100 \times 100$ Pearson
correlation matrices over a 100-region atlas.

| axis | values | count |
|---|---|---:|
| contrast | `co2`, `rest` | 2 |
| processing | `MIRNoise_bold`, `bpfBOLD`, `bpfVASO`, `optcom_bold`, `optcomMIRDenoised_bold` | 5 |
| band | `s4`, `s5`, `sstar` | 3 |
| aggregation | `global` (1) · `inter` per-band (1) · per-patient (×6 subjects) | — |

Layout in `multifunbrain.datasets.april`:

- 10 global aggregates (`co2/rest` × 5 processing, all bands collapsed)
- 30 per-band inter-subject aggregates (`global × band`)
- 180 per-patient × per-band matrices (6 subjects × 2 × 5 × 3)

$\gamma = N/T$ is dataset-dependent: 0.226 for the `bpf*` variants, 0.168 for the
`optcom*` and `MIRNoise` variants. Used to set the MP edge.

Collaborator reference figures: `data/correlation_mat_april_data_results/_example_figs/`

- `freq-contrast-global/{contrast_processing}/{...}_GLOBAL_fig{1,2,3,5}_*.pdf`
  → network heatmap (1), LRG entropy / specific heat (2), brain-volume render (3),
  connectome chord (5). One set per (contrast, processing).
- `freq-contrast-inter/{contrast_processing}/{...}_{band}_fig{1,2,3,5}_*.pdf`
  → same four figures per band.

---

## 2. Headline issue — matrices are dominated by a single positive baseline

All off-diagonal weights cluster in a narrow positive lump; negative
correlations are rare or absent.

| level | $\langle C\rangle_\text{off}$ range | residual $\sigma$ range | frac_neg |
|---|---:|---:|---:|
| global aggregates (10) | 0.25 – 0.92 | 0.07 – 0.21 | 16 % – 53 % |
| per-patient (180) | 0.02 – 0.74 | 0.11 – 0.29 | 0 % – 50 % |

Visible consequences:

- **Edge-weight distributions** (notebook §2): narrow positive lobe, vestigial
  negative lobe, no separation of populations.
- **Raw and MP-cleaned heatmaps** (§3, §4): uniformly bright, no block structure.
  See collaborator file
  `_example_figs/freq-contrast-global/co2_MIRNoise_bold/co2_MIRNoise_bold_GLOBAL_fig1_network.pdf`
  for the extreme case (a saturated near-constant matrix).
- **Percolation** (§5): single sharp drop in $P_\infty(\theta)$, no staircase.
  Direct restatement of "every edge $\approx$ the same weight". The
  collaborator's `..._GLOBAL_fig2_lrg.pdf` shows the same pattern in spectral
  form — the LRG entropy curve is structureless because the leading eigenmode
  dwarfs the rest.
- **Network drawings** (§6): hairballs at any threshold above ~50 % of edges.
- **Verification §7** (per-matrix off-diagonal mean subtraction): residual std is
  small but non-trivial (0.07 – 0.21) — there *is* structure under the baseline,
  it is just dwarfed.

Mechanism (most parsimonious): **upstream pipeline is missing global-signal
regression or proper time-series demeaning**. The whole-brain BOLD mean
contaminates every pairwise correlation. Hypercapnia (CO2) amplifies this
because vasodilation raises every region's BOLD signal together.

---

## 3. Globals are the worst possible view

Cross-subject averaging amplifies the common positive baseline (which adds
coherently) relative to subject-idiosyncratic structure (which is spatially
heterogeneous and partially cancels). Result: globals look 3 – 5 × more
baseline-dominated than the underlying per-patient matrices.

| (contrast, processing) | per-patient median $\langle C\rangle_\text{off}$ | global (one matrix) |
|---|---:|---:|
| co2 · optcomMIRDenoised_bold | **0.081** | 0.401 |
| rest · optcomMIRDenoised_bold | **0.057** | 0.254 |
| co2 · bpfVASO | **0.201** | 0.555 |
| rest · bpfVASO | **0.219** | 0.342 |
| co2 · bpfBOLD | **0.266** | 0.884 |
| rest · bpfBOLD | **0.288** | 0.393 |

Any LRG / connectivity finding read off the collaborator's `freq-contrast-global/`
figures is therefore strongly biased toward "no structure". The per-band
`freq-contrast-inter/` figures are still cross-subject aggregates and inherit a
weaker version of the same amplification.

---

## 4. Variant ranking — by residual structure under the baseline

Lower $\langle C\rangle_\text{off}$ and higher residual $\sigma$ = the variant
suppresses more of the global vascular mode.

| processing | per-pt median $\langle C\rangle_\text{off}$ (co2 / rest) | per-pt median $\sigma_\text{resid}$ | comment |
|---|---:|---:|---|
| **optcomMIRDenoised_bold** | 0.081 / 0.057 | 0.20 / 0.22 | best — multi-echo + denoising actually removes vascular signal |
| **bpfVASO** | 0.201 / 0.219 | 0.19 / 0.18 | second — VASO is intrinsically less vascular |
| bpfBOLD | 0.266 / 0.288 | 0.19 / 0.19 | bandpass alone barely touches the global mode |
| optcom_bold | 0.344 / 0.267 | 0.17 / 0.18 | multi-echo without the denoising stage |
| **MIRNoise_bold** | 0.485 / 0.571 | 0.20 / 0.18 | **useless — `frac_neg = 0 %` across the cohort.** Matrix is constant + ripple |

See `_example_figs/freq-contrast-global/{co2,rest}_MIRNoise_bold/*_fig1_network.pdf`
for the visual signature of the `MIRNoise_bold` collapse.

---

## 5. Band ranking — by residual structure

Per-patient medians across all (subject, contrast, processing):

| band | $\langle C\rangle_\text{off}$ | $\sigma_\text{resid}$ | frac_neg |
|---|---:|---:|---:|
| s4 | 0.196 | 0.171 | 0.117 |
| **s5** | 0.244 | **0.223** | 0.122 |
| sstar | 0.293 | 0.173 | 0.041 |

`s5` carries the most usable residual structure. `sstar` has the highest
baseline and the lowest negative-fraction — the slowest frequency band is also
the most vascular. Worth confirming with collaborators that `sstar` represents
what we think it does; the figure pair
`_example_figs/freq-contrast-inter/rest_optcomMIRDenoised_bold/rest_optcomMIRDenoised_bold_{s5,sstar}_fig1_network.pdf`
makes the contrast visible.

---

## 6. The CO2 vs rest "effect" is vascular at the global level

- Within every processing variant, $\langle C\rangle_\text{off}^{\text{co2}} > \langle C\rangle_\text{off}^{\text{rest}}$ at the global level.
- Maximum gap: `bpfBOLD` 0.88 vs 0.39 (Δ = 0.49).
- Canonical hypercapnia signature: CO2 → vasodilation → whole-brain BOLD ↑ →
  uniform inflation of pairwise correlations. **This is physiology, not
  connectivity reorganisation.**
- At the per-patient × `optcomMIRDenoised_bold` level the gap collapses to
  Δ ≈ 0.02 – 0.07 — within the cleanest preprocessing branch the vascular
  component is partially controlled and a connectivity comparison becomes
  plausible.

---

## 7. Recoverable corner of the cohort

| restriction | rationale |
|---|---|
| `optcomMIRDenoised_bold` | only variant with consistently near-zero baseline |
| band `s5` (or `s4`) | highest residual structure; `sstar` is vascular-dominated |
| **per-patient**, not aggregates | globals erase the structure (§3) |

≈ 12 – 24 matrices (6 subjects × 2 contrasts × {s5} or {s4, s5}). Best
individual examples (lowest $\langle C\rangle_\text{off}$, highest
$\sigma_\text{resid}$):

| subject | band | contrast | $\langle C\rangle_\text{off}$ | $\sigma_\text{resid}$ | frac_neg |
|---|---|---|---:|---:|---:|
| sub-00307729 | s5 | rest | 0.033 | 0.290 | 0.46 |
| sub-VA11266  | s5 | rest | 0.036 | 0.266 | 0.46 |
| sub-VA9757   | s5 | co2  | 0.036 | 0.254 | 0.44 |
| sub-00307729 | s4 | rest | 0.022 | 0.230 | 0.50 |
| sub-VA9757   | s4 | rest | 0.024 | 0.224 | 0.46 |

These look like normal signed FC matrices (mean ≈ 0, balanced positive /
negative population, σ ≈ 0.25). Downstream methods can run on them and report
on connectivity rather than on a global mean.

---

## 8. Weird things / open questions

1. **MIRNoise_bold yields zero negative entries across the entire cohort.** Not
   "few" — exactly zero. Either the denoising kernel rectifies or removes any
   anti-correlated signal, or upstream input differs from the other variants.
   Worth a single sanity check before discarding the variant.
2. **`sstar` is the most vascular-dominated band.** Naive expectation is that
   the lowest-frequency band would carry resting-state structure; instead it
   carries the most slow-vascular drift. Confirm the frequency range each band
   actually represents.
3. **Inter-subject heterogeneity is large**: per-patient
   $\langle C\rangle_\text{off}$ spans 0.02 – 0.74. Either real biology, motion
   / physiology variability, or both. Untriaged.
4. **CO2 / rest labelling caveat** is already in `MEMORY.md` — re-verify per
   subject before any contrast claim.
5. **Missing GSR** is the most parsimonious upstream explanation for the entire
   baseline-dominance picture. Confirm with collaborators. If GSR was applied,
   something else broke (correlation computed without time-series demeaning,
   rectified BOLD, similarity measure not Pearson).

---

## 9. Practical recommendations

1. **One short question to collaborators**: was global-signal regression
   (or equivalent demeaning) applied? Blocking for any connectivity reading.
2. **Lock downstream analyses to the surviving corner** until that is answered:
   `optcomMIRDenoised_bold × s5 × per-patient`.
3. **If forced to use other variants**, deflate by the leading eigenmode
   ($\tilde C = C - \lambda_1 v_1 v_1^\top$), not the scalar mean. Scalar
   subtraction removes one degree of freedom; rank-1 deflation removes the full
   spatially-structured global mode and is the correlation-space analogue of
   GSR.
4. **Concrete next analyses** on the surviving corner:
   - within-subject ARI between band partitions (s4 vs s5 vs sstar) under
     optcomMIRDenoised — reproducibility check
   - within-subject co2 vs rest on rank-1-deflated residuals — original
     research question, restricted to the cleanest slice
   - inter-subject consistency of leading eigenvectors across 6 subjects'
     (rest, optcomMIRDenoised, s5) matrices — population-level signature?
5. **Don't report on the global aggregates.** They are an averaging artefact;
   the collaborator's `_example_figs/freq-contrast-global/` views correspond to
   the worst slice of the dataset.

---

## 10. Figure cross-reference

| collaborator figure | what it shows | how to read it given §2 |
|---|---|---|
| `..._fig1_network.pdf` | heatmap + network drawing | uniform brightness = baseline-dominated; compare `MIRNoise_bold` (worst) to `optcomMIRDenoised_bold` (best) |
| `..._fig2_lrg.pdf` | LRG entropy / specific heat | curves track $\langle C\rangle_\text{off}$; co2 vs rest difference is mostly vascular |
| `..._fig3_brain.pdf` | brain-volume node rendering | spatial map of node-level metric; treat with §7 restriction in mind |
| `..._fig5_connectome.pdf` | connectome chord plot | the "hairball" view in the high-baseline panels is the same observation as §6 of the notebook |

Pairings worth opening side-by-side:

- `freq-contrast-global/co2_bpfBOLD/..._fig1_network.pdf` vs
  `freq-contrast-global/co2_optcomMIRDenoised_bold/..._fig1_network.pdf`
  → variant ranking made visible at the global level.
- `freq-contrast-inter/rest_optcomMIRDenoised_bold/..._{s4,s5,sstar}_fig1_network.pdf`
  → band ranking at the cleanest preprocessing.
- Any `freq-contrast-inter/MIRNoise_bold/...` set → the "constant + ripple"
  failure mode that motivates the open question §8.1.
