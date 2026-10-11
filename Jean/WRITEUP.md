# Predicting incidence rates from a category qualifier

Deliverable 1 is `predictions.csv`. This file answers Deliverables 2 and 3, with supporting
detail in the Appendix. `python solution.py` reproduces every number in about 10 seconds.

---

## Deliverable 2

### How the method works

Incidence is driven mostly by **how common the behaviour is**. The training data cannot
teach that for categories it has never seen, so I supply it as a feature.

- **Target:** `logit(incidence)`. This keeps predictions in (0, 1) and gives intervals
  the right shape (Appendix B).
- **9 inputs, ordinary least squares:**
  - a **world-knowledge prior**: an LLM's estimate of the qualifying share, made from the
    qualifier text only, with no access to any incidence values (`prior/README.md`)
  - **country** (4 dummies, Australia = baseline)
  - **recall window** (log months, plus a has-window flag)
  - whether **"No, but open or likely to…"** qualifies
  - **yes/no vs. multi-select** format
- **Interval:** conformal, using out-of-fold residuals, calibrated separately for
  predicted incidence < 30%, 30–70%, 70–85% and > 85%.

A richer model (TF-IDF text, ridge, boosting) was 0.6 pp better. I chose the 9-input model
because every coefficient can be read and defended (Appendix A).

### How I convinced myself it works: validation

**Split.** Every prediction-set category is unseen, but every country is known. So I hold
out **whole categories in all their countries** with 5-fold GroupKFold, repeated 5 times.
Groups join rows that share a category name *or* an identical qualifier, because 16
qualifiers appear under more than one name. This gives 515 groups.

**Results (grouped-CV MAE):**

| Model | MAE (pp) |
|---|---|
| Global mean | 17.1 |
| Wording + country, no prior | 16.3 |
| Prior only | 7.3 |
| **Prior + country + wording (final)** | **6.7** |

Three further checks:

- **The split matters.** A text model that can memorise categories scores 10.5 pp under a
  random split but 14.3 pp under the grouped split, so a random split would over-promise.
- **The prior is not fragile.** Adding noise to it degrades MAE smoothly, and even a much
  worse prior beats none (Appendix C).
- **The intervals are calibrated.** Calibrating on half the groups and testing on the
  other half gives 80.4% coverage, and 80–81% in every incidence segment.

### Expected error on the prediction set

> **MAE ≈ 6.6 pp (plausible range 5.5–7.5); interval coverage ≈ 80%.**

*Why:* the CV reproduces the prediction-set situation (unseen categories, known
countries), and I re-weighted it to the prediction set's country mix. The range allows for
only about 60 distinct categories, where luck alone moves MAE by about ±1 pp. Coverage may
be lower for US and UK rows (≈ 76–77%) and higher for Australia (≈ 85%).

### What the interval represents

An 80% prediction interval for **the incidence Tracksuit's tracking would report** for
this qualifier.

| Uncertainty source | Share | As we collect more data |
|---|---|---|
| **Model / epistemic:** how common the category is, and how wording shifts it | ≈ 96% | **Shrinks, but slowly.** MAE goes 7.0 → 6.7 pp from 25% to 100% of training categories. Further gains need a better prior, not more rows. |
| **Sampling noise** in the measured rate (median n ≈ 1,600) | ≈ 4% | **Does not shrink with more training data.** It depends on the survey size of each measurement. |

**Not included:**

- A real incidence test of about 220 people adds ≈ 3–4 pp sampling sd. Add
  deff·p(1−p)/n to the variance, which widens the interval by about 8%.
- Drift over time.
- Qualifiers unlike anything in training.

### Limitations, in the order I would tackle them

1. **Approach: the prior is one LLM session's judgement.** Regenerate it via API with a
   fixed prompt and multiple samples, and re-run CV. It drives most of the accuracy, so it
   must be reproducible.
2. **Results: low-incidence categories are under-predicted by ≈ 4 pp, and coverage is
   uneven by country** (US 76%, AU 85%). Fix with a low-end correction and calibration by
   segment × country. This is the region that decides sign-ons.
3. **Approach: output the decision.** Report P(incidence > threshold) and set the rule from
   the cost of a wrong sign-on versus a lost sale.
4. **Data: identical qualifiers in the same country differ by up to 18.5 pp** (55 cases).
   Find out why (base, quotas, panel). If it can be recorded, it becomes a feature and
   lowers the noise floor.
5. **Approach: country effects are the same for every category.** A hierarchical
   country × category-family model would address this.
6. **Data: one wave only, and Canada has 12 rows.** Seasonality, drift and Canadian
   calibration cannot be checked yet.

---

## Deliverable 3: Air Conditioners, United States

> **Prediction ≈ 42%, 80% interval 28–56%.**
> **Trust: moderate in the number, high in the sales decision.**
> P(> 20%) = 0.99, P(> 25%) = 0.95, P(> 30%) = 0.87, P(> 40%) = 0.56.

**How.** Australia measured **34.6%** with the same qualifier, so this is a *transfer*
from a known value, not a prediction from scratch. I added the average AU→US shift of the
8 most similar big-ticket home purchases, which is +0.30 logit (for example, large
appliances 62→65% and flooring 36→41%). Two cross-checks agree within 3–4 pp: all 48
same-wording pairs give 39%, and the general model gives 46% (Appendix D).

**Why the interval is wider than a typical transfer.** About 88% of US households use AC
versus about 74% in Australia. The US mix of central and window units differs. The
fieldwork fell in the US summer but the Australian winter. These structural differences
mostly point upward but cannot be quantified, so I widened the shift uncertainty from
0.33 to 0.45 logit.

**Recommendations.**

1. Clarify the US wording to state which AC types count. Narrowing it to one type could
   lower incidence a lot.
2. Skip the incidence test if the sign-on threshold is ≤ 25%. Run one if it is 30–45%.

---

## Appendix

### A. Fitted model

Coefficients on the logit scale:

- **Prior: 0.86.** Below 1, so the prior is slightly over-extreme and is pulled to the
  middle. The intercept lifts predictions overall, because the prior under-estimates on
  average.
- **Country (vs. Australia):** UK +0.31, Canada +0.18, US +0.16, NZ −0.10. At 50%
  incidence, +0.31 is about +7.7 pp.
- **Wording:** log recall −0.14, future-intent option −0.07, yes/no −0.04. These are small
  because the prior already read the wording, so they only correct it.

**Sample-size weighting was not needed:** sampling noise is 3.6% of residual variance, so
inverse-variance weights are almost uniform.

**Richer model tried:** TF-IDF word and character n-grams, ridge (α by grouped CV), and a
boosted residual correction. MAE was 6.1 pp. Not chosen: it adds thousands of parameters
for 0.6 pp.

### B. Why logit, and why the interval is calibrated per segment

On the raw percentage scale, the same model has the same point accuracy (6.58 vs 6.66 pp;
1 of 833 predictions falls outside [0, 1]). The difference is in the interval.

Real errors (10th–90th percentile of actual − predicted) are:

- **< 30% predicted:** −7 to +10 pp, skewed up because incidence cannot fall below 0%
- **30–70%:** −11 to +11 pp
- **> 85%:** −7 to +4 pp, skewed down because incidence cannot exceed 100%

Coverage by segment, calibrated on half the groups and tested on the other half, 200 times:

| Interval method | All | < 30% | 30–70% | 70–85% | > 85% | Width (pp) |
|---|---|---|---|---|---|---|
| Raw %, one width | 80.0 | 87.5 | 76.2 | 85.8 | 94.9 | 20.9 |
| Raw %, per segment (separate lower/upper) | 78.2 | 73.7 | 79.3 | 76.6 | 75.6 | 20.0 |
| Logit, one width | 80.2 | 83.3 | 82.1 | 77.3 | 68.7 | 21.0 |
| **Logit, per segment (final)** | **80.4** | **81.5** | **80.2** | **80.1** | **81.1** | 21.0 |

- **Raw %, one width:** reaches 80% only because errors cancel. It is too narrow in the
  middle, too wide at the ends, and symmetric where real errors are skewed.
- **Raw %, per segment:** has to learn two quantiles per segment from as few as 49 rows,
  which is too noisy.
- **Logit:** gives the right shape for free but over-compresses above 85%. Per-segment
  widths fix that with one number per segment: q = 0.47, 0.48, 0.53, 0.62.

### C. Validation details

**Leakage demo.** For the 9-input model, a random split gives 6.65 pp versus 6.68 grouped,
because it is too small to memorise categories. For a bag-of-words text model, it is 10.5
versus 14.3.

**Prior robustness.** Gaussian noise added to `logit(prior)`:

| Added noise (sd, logit) | ≈ extra error per rating | MAE (pp) |
|---|---|---|
| 0 | — | 6.7 |
| 0.3 | ± 7 pp | 7.9 |
| 0.6 | ± 15 pp | 10.1 |
| 1.0 | ± 23 pp | 12.4 |
| no prior | — | 16.3 |

**Error by country (OOF MAE, pp):** AU 6.4, UK 6.4, NZ 6.7, US 7.2, CA 7.3. The
prediction-set mix is 38% AU, 29% NZ, 16% US, 15% UK, 2% CA.

**Which validation for which question:**

| Scenario | What is new | Validation |
|---|---|---|
| Prediction set | the category | hold out whole categories |
| AC in the US | this category in this country | AU→US same-category pairs |
| New market (not asked) | the country | hold out a whole country |

**Other data notes:**

- 299 rows have `percentage_3` rounded to 2 dp.
- 31 rows are gender-restricted, and the respondent base is not recorded.
- The design effect of 1.5 is assumed.

### D. Air Conditioners: supporting numbers

| Approach | Estimate |
|---|---|
| General model, ignoring the AU measurement | 46% (80%: 34–59%) |
| AU + mean shift of 48 same-wording pairs (+0.21 logit; US higher in 75%) | 39% (80%: 30–50%) |
| **AU + mean shift of 8 durable pairs (+0.30 logit, sd 0.14)** | **42%** |

The 8 durable pairs: large household appliances, flooring, rugs, premium cookware, new
cars, car modifications, homeware, bonding and sealing.

**Structural factors:**

- **Installed base:** about 88% of US households use AC
  ([EIA RECS 2020](https://www.eia.gov/todayinenergy/detail.php?id=52558)), versus about
  74% of Australian households with any cooling
  ([ABS 2014](https://www.abs.gov.au/ausstats/abs@.nsf/Lookup/4602.0.55.001main+features1Mar%202014)).
- **Product mix:** about two-thirds of US homes have central AC, which is replaced rarely.
  Window and portable units are cheap and bought often. Australian reverse-cycle splits
  also heat.
- **Renters:** about a third of US households rent, and they mostly buy window units.

**Suggested US wording:** "…an air conditioner or air conditioning system for your home?
(Note: includes central air, ductless/mini-split, window, wall and portable units.)"

Approaches B and C do not use the prior, which was rated after the AU value had been seen.
