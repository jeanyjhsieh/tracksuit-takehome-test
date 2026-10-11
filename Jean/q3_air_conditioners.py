"""Deliverable 3: Air Conditioners in the United States.

    python q3_air_conditioners.py

A. The general model, treating AC as an unseen US category.
B. Transfer from the observed Australian rate using all AU-US same-wording pairs.
C. Transfer using the closest analogues: big-ticket home / durable purchases.
"""
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from solution import train, X, y, features, FINAL, oof_predictions, N_REPEATS

DEFF = 1.5
au = train[(train.category_name == "Air Conditioners") & (train.geography_name == "Australia")].iloc[0]
p_au, n_au = au.percentage_3, au.base_weight_3
print(f"Observed AU: {p_au:.3f} (n={n_au:.0f})\n  Q: {au.question_text}\n  qualifying: {au.qualifying_options}\n")

# ---------------------------------------------------------------- A. general model
oof = np.mean([oof_predictions(FINAL, r) for r in range(N_REPEATS)], axis=0)
q = np.quantile(np.abs(y - oof), 0.8)
model = LinearRegression().fit(X[FINAL], y)
us = pd.DataFrame([au]).assign(geography_name="United States", prior=0.40)  # AU rating reused
z = model.predict(features(us)[FINAL])[0]
print(f"A. general model (same prior as AU): {expit(z):.3f}  80% [{expit(z-q):.3f}, {expit(z+q):.3f}]")

# ---------------------------------------------------------------- B. all same-wording pairs
k = ["category_name", "question_text", "qualifying_options"]
pairs = (train[train.geography_name == "Australia"]
         .merge(train[train.geography_name == "United States"], on=k, suffixes=("_au", "_us"))
         .groupby(k, as_index=False).first())
shift = logit(pairs.percentage_3_us) - logit(pairs.percentage_3_au)
samp = DEFF / (pairs.base_weight_3_au * pairs.percentage_3_au * (1 - pairs.percentage_3_au)) + \
       DEFF / (pairs.base_weight_3_us * pairs.percentage_3_us * (1 - pairs.percentage_3_us))
true_sd = np.sqrt(shift.var() - samp.mean())          # remove sampling noise from the spread
samp_au = DEFF / (n_au * p_au * (1 - p_au))
samp_us = DEFF / (1600 * 0.4 * 0.6)                    # typical future US measurement
sd_B = np.sqrt(true_sd ** 2 + samp_au + samp_us)
zB = logit(p_au) + shift.mean()
print(f"B. {len(pairs)} AU-US same-wording pairs: mean shift {shift.mean():+.3f} logit, "
      f"true-shift sd {true_sd:.3f}, US>AU in {(shift > 0).mean():.0%}")
print(f"   -> {expit(zB):.3f}  80% [{expit(zB-1.2816*sd_B):.3f}, {expit(zB+1.2816*sd_B):.3f}]")

# ---------------------------------------------------------------- C. home / durable analogues
DURABLES = ["Large Household Appliances", "Flooring", "Rugs", "Premium Cookware",
            "New Cars", "Car Modifications", "Homeware", "Bonding and Sealing"]
by_cat = lambda g: train[train.geography_name == g].groupby("category_name").percentage_3.first()
dur = pd.DataFrame({"AU": by_cat("Australia")[DURABLES], "US": by_cat("United States")[DURABLES]})
dur["shift"] = logit(dur.US) - logit(dur.AU)
print("\nC. durable analogues\n", dur.round(3).to_string())
zC = logit(p_au) + dur["shift"].mean()
print(f"   mean shift {dur['shift'].mean():+.3f} (sd {dur['shift'].std():.3f}) -> {expit(zC):.3f}")

# ---------------------------------------------------------------- combined
# Centre on C; widen the shift sd from the typical ~0.33 to 0.45 logit because AC has
# structural country drivers (installed base, central vs window units, season, renters).
SD = np.sqrt(0.45 ** 2 + samp_au + samp_us)
print(f"\nCOMBINED: {expit(zC):.3f}  80% [{expit(zC-1.2816*SD):.3f}, {expit(zC+1.2816*SD):.3f}]")
for thr in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    print(f"   P(US incidence > {thr:.0%}) = {1 - norm.cdf((logit(thr) - zC) / SD):.2f}")
