"""Tracksuit incidence-rate prediction: simple, end-to-end.

    python solution.py

Model: linear regression on logit(incidence) with
    - a world-knowledge prior (how common is this behaviour?)   -> prior/
    - geography
    - four qualifier-wording features
Validation: repeated GroupKFold over leakage-safe groups of categories.
Interval: conformal on out-of-fold logit residuals, calibrated per segment of
          predicted incidence (<30%, 30-70%, 70-85%, >85%).
Writes predictions.csv and prints every number quoted in WRITEUP.md.
"""
import re
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold

SEED, N_REPEATS, N_FOLDS, LEVEL = 0, 5, 5, 0.80
GEOS = ["New Zealand", "United Kingdom", "United States", "Canada"]   # Australia = baseline


# =============================================================== data + prior
def attach_prior(df):
    """Join the LLM's text-only estimate of the qualifying share (see prior/README.md)."""
    keys = pd.read_csv("prior_keys.csv")
    rat = pd.read_csv("prior/prior_ratings.csv", header=None, names=["key", "prior_pct"])
    k = ["geography_name", "question_text", "qualifying_options"]
    out = df.merge(keys.merge(rat, on="key")[k + ["prior_pct"]], on=k, how="left")
    assert len(out) == len(df) and out.prior_pct.notna().all()
    out["prior"] = out.prior_pct / 100
    return out.drop(columns="prior_pct")


train = attach_prior(pd.read_csv("incidence_dataset.csv"))
test = attach_prior(pd.read_csv("prediction_set.csv"))
y = logit(train.percentage_3.clip(0.005, 0.995)).values


# =============================================================== features
_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "ten": 10, "twelve": 12}


def recall_months(text):
    """'In the past 3 months…' -> 3; 'past 2 years' -> 24; no window -> NaN."""
    m = re.search(r"(past|last|previous)\s+(\d+|one|two|three|four|five|six|ten|twelve)?\s*"
                  r"(day|week|month|year)s?", text.lower())
    if not m:
        return np.nan
    n = m.group(2)
    n = 1 if n is None else (int(n) if n.isdigit() else _NUM[n])
    return n * {"day": 1 / 30, "week": 0.25, "month": 1, "year": 12}[m.group(3)]


def options(s):
    return [o.strip() for o in str(s).split("|") if o.strip()]


def is_future_intent(o):
    """'No, but I am open or likely to…', 'I intend to…', 'Thought about…'"""
    return bool(re.search(r"\bno\b.*(but|open|likely|plan|hoping|intend)|open or likely|"
                          r"intend to|planning to|thought about|considering", o.lower()))


def features(df):
    X = pd.DataFrame(index=df.index)
    X["prior_logit"] = logit(df.prior.clip(0.02, 0.98))
    for g in GEOS:
        X[f"geo_{g}"] = (df.geography_name == g).astype(float)
    rm = df.question_text.map(recall_months)
    X["has_recall"] = rm.notna().astype(float)
    X["log_recall_months"] = np.log(rm.fillna(12))       # 12 = neutral fill, flagged above
    qual = df.qualifying_options.map(options)
    X["future_intent_qualifies"] = qual.map(lambda os: any(map(is_future_intent, os))).astype(float)
    X["yes_no_format"] = qual.map(lambda os: all(o.lower().startswith(("yes", "no"))
                                                 for o in os)).astype(float)
    return X


X, X_test = features(train), features(test)


# =============================================================== leakage-safe groups
def leakage_groups(df):
    """Rows sharing a category name OR an identical qualifier (question + qualifying
    options) end up in the same group (connected components, union-find)."""
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for c, q, o in zip(df.category_name, df.question_text, df.qualifying_options):
        parent[find("C:" + c.strip().lower())] = find("Q:" + q.strip().lower() + "|" + str(o).lower())
    return np.array([find("C:" + c.strip().lower()) for c in df.category_name])


groups = leakage_groups(train)


# =============================================================== cross-validation
def oof_predictions(cols, seed, splitter="group", Xm=None, target=None):
    """Out-of-fold predictions for one repeat (default: logit target, standard features)."""
    Xm = X if Xm is None else Xm
    target = y if target is None else target
    oof = np.zeros(len(train))
    if splitter == "group":
        u = np.unique(groups)
        perm = dict(zip(u, np.random.RandomState(seed).permutation(len(u))))
        split = GroupKFold(N_FOLDS).split(X, y, np.array([perm[g] for g in groups]))
    else:
        split = KFold(N_FOLDS, shuffle=True, random_state=seed).split(X)
    for tr, va in split:
        if not cols:                                    # baseline: global mean
            oof[va] = target[tr].mean()
            continue
        m = LinearRegression().fit(Xm.iloc[tr][cols], target[tr])
        oof[va] = m.predict(Xm.iloc[va][cols])
    return oof


def mae(oof_logit):
    return np.abs(expit(oof_logit) - train.percentage_3.values).mean()


GEO_COLS = [f"geo_{g}" for g in GEOS]
WORDING = ["has_recall", "log_recall_months", "future_intent_qualifies", "yes_no_format"]
FINAL = ["prior_logit"] + GEO_COLS + WORDING

# Interval: conformal, calibrated separately within segments of predicted incidence
# ("Mondrian" conformal). Logit gives the shape; segments correct how much it shrinks
# near the bounds. Segment edges are coarse and keep >= ~50 training rows per segment.
SEG_EDGES = [0.30, 0.70, 0.85]
SEG_NAMES = ["<30%", "30-70%", "70-85%", ">85%"]


def segment(pred_logit):
    return np.digitize(expit(pred_logit), SEG_EDGES)


def conformal_q(scores, level=LEVEL):
    k = int(np.ceil((len(scores) + 1) * level))
    return np.sort(scores)[min(k, len(scores)) - 1]


def segment_q(abs_resid, seg):
    return np.array([conformal_q(abs_resid[seg == s]) for s in range(len(SEG_NAMES))])

if __name__ == "__main__":
    print(f"{len(train)} rows, {train.category_name.nunique()} categories, "
          f"{len(np.unique(groups))} leakage-safe groups\n")

    # ---- 1. model comparison (grouped CV, 5 repeats)
    print("Grouped CV MAE (pp), mean ± sd over repeats")
    for name, cols in [("global mean", []),
                       ("wording + geography, no prior", GEO_COLS + WORDING),
                       ("prior only", ["prior_logit"]),
                       ("prior + geography", ["prior_logit"] + GEO_COLS),
                       ("FINAL: prior + geography + wording", FINAL)]:
        s = [mae(oof_predictions(cols, r)) for r in range(N_REPEATS)]
        print(f"  {name:<38} {100*np.mean(s):5.2f} ± {100*np.std(s):.2f}")
    rnd = np.mean([mae(oof_predictions(FINAL, r, "random")) for r in range(N_REPEATS)])
    print(f"  {'FINAL, but RANDOM row split':<38} {100*rnd:5.2f}   <- ~same: 9 params can't memorise")

    # Why grouping still matters: a model that CAN memorise categories (bag-of-words on
    # the qualifier text) looks far better under a random split than it really is.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    txt = train.category_name + " " + train.question_text + " " + train.qualifying_options.astype(str)
    for split_name, split in [("grouped", GroupKFold(N_FOLDS).split(X, y, groups)),
                              ("random", KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X))]:
        o = np.zeros(len(train))
        for tr, va in split:
            v = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit(txt.iloc[tr])
            o[va] = Ridge(1.0).fit(v.transform(txt.iloc[tr]), y[tr]).predict(v.transform(txt.iloc[va]))
        print(f"  {'text-only model, ' + split_name + ' split':<38} {100*mae(o):5.2f}")
    print()

    # ---- 2. out-of-fold predictions of the final model (averaged over repeats)
    oof = np.mean([oof_predictions(FINAL, r) for r in range(N_REPEATS)], axis=0)
    resid = y - oof
    err = np.abs(expit(oof) - train.percentage_3.values)
    print(f"OOF: MAE {100*err.mean():.2f}pp, median {100*np.median(err):.2f}, "
          f"RMSE {100*np.sqrt((err**2).mean()):.2f}, 90th pct {100*np.quantile(err, .9):.2f}")
    by_geo = pd.Series(err).groupby(train.geography_name).mean()
    mix = test.geography_name.value_counts(normalize=True)
    print("OOF MAE by geography (pp):", (100 * by_geo).round(1).to_dict())
    print(f"Expected MAE on prediction set (geo-weighted): {100*(mix*by_geo).sum():.2f}pp")
    print("Mean residual by geography (logit):",
          pd.Series(resid).groupby(train.geography_name).mean().round(3).to_dict())

    # sampling noise vs total residual variance
    p, n = train.percentage_3.clip(0.005, 0.995), train.base_weight_3
    samp_var = 1.5 / (n * p * (1 - p))                     # delta method, design effect 1.5
    print(f"Sampling noise share of residual variance: {samp_var.mean()/resid.var():.1%}\n")

    # ---- 3. interval
    seg = segment(oof)
    print("training rows per predicted-incidence segment:",
          dict(zip(SEG_NAMES, np.bincount(seg, minlength=4))))

    # Same model on the raw percentage scale (prior as a raw share), clipped to [0, 1]
    Xr = X.copy()
    Xr["prior_logit"] = train.prior.values
    p_obs = train.percentage_3.values
    oof_raw = np.clip(np.mean([oof_predictions(FINAL, r, Xm=Xr, target=p_obs)
                               for r in range(N_REPEATS)], axis=0), 0.001, 0.999)
    print(f"point estimates: raw-% scale MAE {100*np.abs(oof_raw-p_obs).mean():.2f}pp vs "
          f"logit {100*err.mean():.2f}pp; raw predictions outside [0,1]: "
          f"{int(((oof_raw <= 0.001) | (oof_raw >= 0.999)).sum())}")

    def bounds(method, cal, ev):
        if method == "raw %, constant width":
            q = conformal_q(np.abs(p_obs - oof_raw)[cal])
            return np.clip(oof_raw[ev] - q, 0, 1), np.clip(oof_raw[ev] + q, 0, 1)
        if method == "raw %, per segment (asym.)":
            lo, hi = np.zeros(ev.sum()), np.zeros(ev.sum())
            for k in range(4):
                r, e = (p_obs - oof_raw)[cal & (seg == k)], seg[ev] == k
                lo[e] = np.clip(oof_raw[ev][e] + np.quantile(r, 0.1), 0, 1)
                hi[e] = np.clip(oof_raw[ev][e] + np.quantile(r, 0.9), 0, 1)
            return lo, hi
        if method == "logit, constant width":
            q = np.full(ev.sum(), conformal_q(np.abs(resid[cal])))
        else:  # "logit, per segment" (final)
            q = segment_q(np.abs(resid[cal]), np.where(cal, seg, -1)[cal])[seg[ev]]
        return expit(oof[ev] - q), expit(oof[ev] + q)

    # honest check: calibrate on half the groups, evaluate on the other half, 200 times
    METHODS = ["raw %, constant width", "raw %, per segment (asym.)",
               "logit, constant width", "logit, per segment"]
    GEO4 = ["Australia", "New Zealand", "United Kingdom", "United States"]
    rng, u, rows = np.random.RandomState(1), np.unique(groups), []
    for _ in range(200):
        cal = np.isin(groups, rng.choice(u, len(u) // 2, replace=False))
        ev = ~cal
        for mth in METHODS:
            lo, hi = bounds(mth, cal, ev)
            inside = (p_obs[ev] >= lo) & (p_obs[ev] <= hi)
            rows.append({"method": mth, "all": inside.mean(), "width": (hi - lo).mean(),
                         **{n_: inside[seg[ev] == k].mean() for k, n_ in enumerate(SEG_NAMES)},
                         **{g: inside[train.geography_name.values[ev] == g].mean() for g in GEO4}})
    cov = pd.DataFrame(rows).groupby("method", sort=False).mean()
    print("\nHeld-out 80% interval coverage / width (%), 200 group splits")
    print((100 * cov).round(1).to_string())
    sd_all = pd.DataFrame(rows).query("method == 'logit, per segment'")["all"].std()
    print(f"sd of overall coverage (final method): {100*sd_all:.1f}")

    q_seg = segment_q(np.abs(resid), seg)
    print("final per-segment q (logit):", dict(zip(SEG_NAMES, q_seg.round(3))), "\n")

    # ---- 4. how does error change with more training categories? (learning curve)
    rng = np.random.RandomState(0)
    for frac in [0.25, 0.5, 0.75, 1.0]:
        e = []
        for r in range(5):
            for tr, va in GroupKFold(N_FOLDS).split(X, y, groups):
                keep = np.isin(groups[tr], rng.choice(np.unique(groups[tr]),
                                                      int(frac * len(np.unique(groups[tr]))), replace=False))
                m = LinearRegression().fit(X.iloc[tr[keep]][FINAL], y[tr[keep]])
                e.append(np.abs(expit(m.predict(X.iloc[va][FINAL])) - train.percentage_3.values[va]).mean())
        print(f"learning curve: {frac:4.0%} of training groups -> MAE {100*np.mean(e):.2f}pp")

    # ---- 5. robustness: how much does the result depend on the prior's quality?
    key = pd.factorize(train.geography_name + train.question_text + train.qualifying_options.astype(str))[0]
    base_prior = X.prior_logit.copy()
    for sd in [0.3, 0.6, 1.0]:
        s = []
        for r in range(3):
            X["prior_logit"] = base_prior + np.random.RandomState(10 + r).normal(0, sd, key.max() + 1)[key]
            s.append(mae(oof_predictions(FINAL, r)))
        print(f"prior + noise sd {sd} logit -> MAE {100*np.mean(s):.2f}pp")
    X["prior_logit"] = base_prior

    # ---- 6. fit on everything, write predictions
    model = LinearRegression().fit(X[FINAL], y)
    print("\ncoefficients (logit scale):",
          dict(zip(FINAL, model.coef_.round(3))), "intercept", round(model.intercept_, 3))
    z = model.predict(X_test[FINAL])
    q_test = q_seg[segment(z)]
    print("prediction rows per segment:", dict(zip(SEG_NAMES, np.bincount(segment(z), minlength=4))))
    pred = pd.DataFrame({"prediction_id": test.prediction_id,
                         "point_estimate": expit(z).round(4),
                         "lower_80": expit(z - q_test).round(4),
                         "upper_80": expit(z + q_test).round(4)})
    assert len(pred) == 110 and pred.prediction_id.is_unique
    pred.to_csv("predictions.csv", index=False)
    print(f"wrote predictions.csv; mean interval width {(pred.upper_80 - pred.lower_80).mean():.3f}")
