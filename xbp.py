#!/usr/bin/env python3
"""XBP: an eXplainable, cost-aware framework for Business Performance prediction.

Reproduces every table and figure of the short communication
"An Explainable Artificial Intelligence Framework for Data Analytics-Driven
Business Performance Prediction" (submitted to Knowledge-Based Systems).

    python xbp.py            # full study: 5-fold x 3-repeat nested CV (~11 min, 2 CPUs)
    python xbp.py --quick    # smoke test: 3-fold x 1 repeat (~2 min)

The four layers of the framework:
  L1  Data analytics   constant-feature removal and Spearman redundancy filtering
  L2  AI prediction    class-weighted gradient boosting (LightGBM)
  L3  Explainability   SHAP ranks the drivers; the top-k drivers form a compact model
  L4  Decision         a cost-optimal threshold learned from inner out-of-fold scores

Every layer is fitted on the outer training fold only, so no information from
the test fold reaches feature filtering, driver selection or threshold choice.
L4 is compared with SMOTE resampling and cost-sensitive training, and per-fold
costs are compared with the Nadeau-Bengio corrected resampled t-test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from matplotlib.colors import LinearSegmentedColormap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from scipy.stats import t as student_t
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import QuantileTransformer
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

TARGET = "Bankrupt?"
DATA_SHA256 = "67bf2e7c75490f7ad3f76bbce57d49cdc25967cdab607527b94f944863fa14d8"
PROPOSED = "XBP (proposed)"
INK, MUTED, BLUE, ORANGE, NEUTRAL = "#0b0b0b", "#52514e", "#1c5cab", "#d95926", "#f0efec"


# ---------------------------------------------------------------- L1: data analytics
def load_data(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the UCI Taiwanese Bankruptcy Prediction data and verify its checksum."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != DATA_SHA256:
        warnings.warn(f"{path} checksum differs from the published dataset ({digest[:12]}...)")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    y = df.pop(TARGET).to_numpy(dtype=int)
    return df.astype(float), y


class RedundancyFilter(BaseEstimator, TransformerMixin):
    """Drop constant features, then one of every pair with |Spearman rho| above `threshold`.

    Columns are visited in their original order, so the first member of a
    redundant pair is kept. Fitted on training data only.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        cols = list(X.columns[X.nunique() > 1])
        rho = X[cols].rank().corr().abs().to_numpy()  # Spearman = Pearson on ranks
        dropped: set[int] = set()
        for i in range(len(cols)):
            if i in dropped:
                continue
            dropped.update(j for j in range(i + 1, len(cols)) if rho[i, j] > self.threshold)
        self.columns_ = [c for k, c in enumerate(cols) if k not in dropped]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(X)[self.columns_]


class Select(BaseEstimator, TransformerMixin):
    """Keep a fixed list of columns (the SHAP-selected drivers)."""

    def __init__(self, columns: tuple[str, ...] = ()):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(X)[list(self.columns)]


# ---------------------------------------------------------------- L2: AI prediction
def lightgbm(pos_weight: float, seed: int, n_jobs: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=pos_weight, random_state=seed, n_jobs=n_jobs, verbose=-1,
    )


def baselines(pos_weight: float, seed: int, n_jobs: int) -> dict[str, Pipeline]:
    """Benchmark models. Hyper-parameters are fixed a priori (no test-set tuning)."""
    scale = lambda: QuantileTransformer(output_distribution="normal", random_state=seed)
    return {
        "Logistic regression": make_pipeline(
            RedundancyFilter(), scale(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)),
        "Multilayer perceptron": make_pipeline(
            RedundancyFilter(), scale(),
            MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                          max_iter=500, random_state=seed)),
        "Random forest": make_pipeline(
            RedundancyFilter(),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=n_jobs,
                                   class_weight="balanced_subsample", random_state=seed)),
        "XGBoost": make_pipeline(
            RedundancyFilter(),
            XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, scale_pos_weight=pos_weight, tree_method="hist",
                          eval_metric="logloss", n_jobs=n_jobs, random_state=seed)),
        "LightGBM": make_pipeline(RedundancyFilter(), lightgbm(pos_weight, seed, n_jobs)),
    }


# ---------------------------------------------------------------- L3: explainability
def shap_ranking(fitted: Pipeline, X: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| of each feature for a fitted filter -> tree-model pipeline."""
    Z = fitted[:-1].transform(X)
    values = shap.TreeExplainer(fitted[-1]).shap_values(Z)
    values = values[1] if isinstance(values, list) else values
    return pd.Series(np.abs(values).mean(axis=0), index=Z.columns).sort_values(ascending=False)


def consensus_ranking(models: list[tuple[Pipeline, pd.DataFrame]]) -> pd.Series:
    """Consensus SHAP ranking: average each model's normalized mean |SHAP| share.

    Motivation: one model's SHAP ranking can split credit arbitrarily between
    correlated ratios. Averaging shares over models fitted on different resamples
    tests whether that model-level noise drives driver instability. (In the paper's
    study it does not: stability equals that of single-model SHAP, so instability
    across folds is sample-driven; the single-model ranking is kept as an ablation.)
    Features a model did not keep count as zero for that model.
    """
    shares = [r / r.sum() for r in (shap_ranking(m, X) for m, X in models)]
    return pd.concat(shares, axis=1).fillna(0.0).mean(axis=1).sort_values(ascending=False)


def jaccard_stability(sets: list[set[str]]) -> float:
    """Mean pairwise Jaccard similarity of the driver sets chosen in different folds."""
    pairs = list(combinations(sets, 2))
    return float(np.mean([len(a & b) / len(a | b) for a, b in pairs])) if pairs else 1.0


# ---------------------------------------------------------------- L4: decision layer
def cost_optimal_threshold(y: np.ndarray, p: np.ndarray, ratio: float) -> float:
    """Threshold t minimizing ratio*FN + FP when firms with p >= t are flagged.

    Exact search over every distinct score (O(n log n)); ratio = cost(FN) / cost(FP).
    """
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    tp = np.concatenate([[0], np.cumsum(ys)])            # flagging the top-k firms
    k = np.arange(len(ps) + 1)
    cost = ratio * (ys.sum() - tp) + (k - tp)
    valid = np.ones(len(cost), bool)
    valid[1:-1] = ps[:-1] > ps[1:]                       # cut only between distinct scores
    best = int(np.argmin(np.where(valid, cost, np.inf)))
    if best == 0:
        return float(np.nextafter(ps[0], np.inf))        # flag nobody
    if best == len(ps):
        return float(ps[-1])                             # flag everybody
    return float((ps[best - 1] + ps[best]) / 2)


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float, ratio: float) -> dict[str, float]:
    flag = p >= threshold
    tp, fp = int(np.sum(flag & (y == 1))), int(np.sum(flag & (y == 0)))
    fn, tn = int(np.sum(~flag & (y == 1))), int(np.sum(~flag & (y == 0)))
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "recall": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "cost": 1000 * (ratio * fn + fp) / len(y),       # per 1,000 firms, in FP-cost units
    }


def fit_with_threshold(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, ratios,
                       inner: StratifiedKFold):
    """Learn cost-optimal thresholds from inner out-of-fold scores, then refit on all of X.

    Returns the refitted model, the thresholds per cost ratio and the inner
    models (reused by L3 to build a consensus SHAP ranking at no extra cost).
    """
    oof, inner_models = np.zeros(len(y)), []
    for a, b in inner.split(X, y):
        m = clone(pipe).fit(X.iloc[a], y[a])
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]
        inner_models.append((m, X.iloc[a]))
    thresholds = {r: cost_optimal_threshold(y, oof, r) for r in ratios}
    return clone(pipe).fit(X, y), thresholds, inner_models


# ---------------------------------------------------------------- experiment
def run(args) -> pd.DataFrame:
    X, y = load_data(args.data)
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    ratios = sorted({*args.sensitivity, args.cost_ratio})
    outer = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats,
                                    random_state=args.seed)
    rows = []
    drivers = {("consensus", k): [] for k in args.k} | {("single", args.k_main): []}
    print(f"{len(y)} firms, {X.shape[1]} ratios, {y.sum()} failures | "
          f"{args.folds}x{args.repeats} CV | cost ratios {ratios}")

    for fold, (tr, te) in enumerate(outer.split(X, y)):
        t0 = time.time()
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed + fold)

        def record(name: str, model, thresholds, n_features: int):
            p = model.predict_proba(Xte)[:, 1]
            for r in ratios:
                rows.append({"fold": fold, "model": name, "ratio": r, "features": n_features,
                             "threshold": "cost-aware", **evaluate(yte, p, thresholds[r], r)})
                rows.append({"fold": fold, "model": name, "ratio": r, "features": n_features,
                             "threshold": "default 0.5", **evaluate(yte, p, 0.5, r)})

        for name, pipe in baselines(pos_weight, args.seed, args.jobs).items():
            model, thr, inner_models = fit_with_threshold(pipe, Xtr, ytr, ratios, inner)
            record(name, model, thr, len(model[0].columns_))
            if name == "LightGBM":                       # L3 input: training-fold models only
                lgbm_models = [(model, Xtr), *inner_models]

        # decision strategies on the same filtered ratios (Table 2)
        smote = make_imb_pipeline(RedundancyFilter(), SMOTE(random_state=args.seed),
                                  lightgbm(1.0, args.seed, args.jobs))   # resampling instead of weights
        model, thr, _ = fit_with_threshold(smote, Xtr, ytr, ratios, inner)
        record("SMOTE + LightGBM", model, thr, len(model[0].columns_))
        for r in ratios:  # predict-and-optimize: failures weighted by c, Bayes-optimal threshold 0.5
            cs = make_pipeline(RedundancyFilter(), lightgbm(r, args.seed, args.jobs)).fit(Xtr, ytr)
            rows.append({"fold": fold, "model": "Cost-sensitive LightGBM", "ratio": r,
                         "features": len(cs[0].columns_), "threshold": "default 0.5",
                         **evaluate(yte, cs.predict_proba(Xte)[:, 1], 0.5, r)})

        rankings = {"consensus": consensus_ranking(lgbm_models),
                    "single": shap_ranking(*lgbm_models[0])}   # ablation: one model's SHAP
        for source, k in [("consensus", k) for k in args.k] + [("single", args.k_main)]:
            top = tuple(rankings[source].index[:k])
            drivers[(source, k)].append(set(top))
            pipe = make_pipeline(Select(top), lightgbm(pos_weight, args.seed, args.jobs))
            model, thr, _ = fit_with_threshold(pipe, Xtr, ytr, ratios, inner)
            record(f"XBP top-{k}" if source == "consensus" else f"Single-SHAP top-{k}",
                   model, thr, k)
        print(f"  fold {fold + 1:2d}/{outer.get_n_splits()}  {time.time() - t0:5.1f}s")

    folds = pd.DataFrame(rows)
    folds.loc[folds.model == f"XBP top-{args.k_main}", "model"] = PROPOSED
    stability = {f"{s}-top-{k}": jaccard_stability(d) for (s, k), d in drivers.items()}
    chosen = drivers[("consensus", args.k_main)]
    frequency = pd.Series([f for s in chosen for f in s]).value_counts()
    return folds, stability, frequency / len(chosen)


def mean_sd(s: pd.Series, digits: int = 3) -> str:
    return f"{s.mean():.{digits}f} ± {s.std(ddof=1):.{digits}f}"


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    order = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for i, name in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvalues[name]))
        adjusted[name] = running
    return adjusted


def corrected_ttest(diff: np.ndarray, test_over_train: float) -> tuple[float, float, float, float]:
    """Nadeau-Bengio corrected resampled t-test for repeated k-fold cross-validation.

    Overlapping training sets make fold scores correlated, so a plain paired test is
    too liberal; the correction inflates the variance by (1/J + n_test/n_train).
    Returns the mean difference, its 95% confidence interval and the two-sided p-value.
    """
    J, mean = len(diff), float(np.mean(diff))
    se = float(np.sqrt((1 / J + test_over_train) * np.var(diff, ddof=1)))
    if se == 0:
        return mean, mean, mean, 1.0
    half = float(student_t.ppf(0.975, J - 1)) * se
    return mean, mean - half, mean + half, float(2 * student_t.sf(abs(mean / se), J - 1))


def fold_cost(folds: pd.DataFrame, model: str, thr: str, r: float) -> np.ndarray:
    g = folds[(folds.model == model) & (folds.threshold == thr) & (folds.ratio == r)]
    return g.sort_values("fold").cost.to_numpy()


def table1(folds: pd.DataFrame, ratio: float, n_folds: int) -> pd.DataFrame:
    """Benchmark comparison at the main cost ratio, all models using cost-aware thresholds."""
    d = folds[(folds.ratio == ratio) & (folds.threshold == "cost-aware")]
    names = ["Logistic regression", "Multilayer perceptron", "Random forest",
             "XGBoost", "LightGBM", PROPOSED]
    ours = fold_cost(folds, PROPOSED, "cost-aware", ratio)
    tests = {n: corrected_ttest(ours - fold_cost(folds, n, "cost-aware", ratio), 1 / (n_folds - 1))
             for n in names[:-1]}
    p_adj = holm({n: t[3] for n, t in tests.items()})
    out = []
    for n in names:
        g = d[d.model == n]
        diff = "—" if n == PROPOSED else "{:+.1f} [{:+.1f}, {:+.1f}]".format(*tests[n][:3])
        out.append({"Model": n, "Ratios": int(round(g.features.mean())),
                    "ROC-AUC": mean_sd(g.roc_auc), "PR-AUC": mean_sd(g.pr_auc),
                    "Recall": mean_sd(g.recall), "Cost per 1,000 firms": mean_sd(g.cost, 1),
                    "XBP − model [95% CI]": diff,
                    "Holm p": "—" if n == PROPOSED else f"{p_adj[n]:.3f}"})
    return pd.DataFrame(out)


def table2(folds: pd.DataFrame, k_values, k_main: int, stability: dict, ratio: float) -> pd.DataFrame:
    """Decision strategies, ablation of the XBP layers, and sensitivity to the cost ratio."""
    name = lambda k: PROPOSED if k == k_main else f"XBP top-{k}"
    variants = [("LightGBM", "default 0.5", "L1 + L2, default 0.5 threshold", None),
                ("LightGBM", "cost-aware", "+ L4 cost-aware threshold", None),
                ("SMOTE + LightGBM", "cost-aware", "SMOTE + cost-aware threshold", None),
                ("Cost-sensitive LightGBM", "default 0.5", "Cost-sensitive training (weight = c)", None)]
    for k in sorted(k_values, reverse=True):
        if k == k_main:
            variants.append((f"Single-SHAP top-{k}", "cost-aware",
                             f"+ L3 single-model SHAP, top {k}", f"single-top-{k}"))
        variants.append((name(k), "cost-aware", f"+ L3 consensus SHAP, top {k}"
                         + (" (XBP)" if k == k_main else ""), f"consensus-top-{k}"))
    out = []
    for model, thr, label, key in variants:
        g = folds[(folds.model == model) & (folds.threshold == thr)]
        row = {"Variant": label, "Ratios": int(round(g.features.mean())),
               "PR-AUC": mean_sd(g[g.ratio == ratio].pr_auc),
               "Driver stability (Jaccard)": f"{stability[key]:.2f}" if key else "—"}
        for r in sorted(folds.ratio.unique()):
            row[f"Cost, c = {r:g}"] = mean_sd(g[g.ratio == r].cost, 1)
        out.append(row)
    return pd.DataFrame(out)


def paired_tests(folds: pd.DataFrame, k_main: int, n_folds: int) -> dict:
    """Corrected resampled t-tests behind the claims in the text (paired by outer fold)."""
    frac = 1 / (n_folds - 1)
    def test(a, b):  # mean of a - b
        m, lo, hi, p = corrected_ttest(a - b, frac)
        return {"mean_diff": round(m, 1), "ci95": [round(lo, 1), round(hi, 1)], "p": round(p, 4)}
    keys = ("threshold_effect_lightgbm", "consensus_vs_single", "threshold_vs_smote",
            "threshold_vs_cost_sensitive_training")
    out = {k: {} for k in keys}
    for r in sorted(folds.ratio.unique()):
        c = f"{r:g}:1"
        default, aware = fold_cost(folds, "LightGBM", "default 0.5", r), fold_cost(folds, "LightGBM", "cost-aware", r)
        out["threshold_effect_lightgbm"][c] = {
            "default": round(default.mean(), 1), "cost_aware": round(aware.mean(), 1),
            "reduction_pct": round(100 * (1 - aware.mean() / default.mean()), 1), **test(default, aware)}
        out["consensus_vs_single"][c] = test(fold_cost(folds, PROPOSED, "cost-aware", r),
                                             fold_cost(folds, f"Single-SHAP top-{k_main}", "cost-aware", r))
        out["threshold_vs_smote"][c] = test(aware, fold_cost(folds, "SMOTE + LightGBM", "cost-aware", r))
        out["threshold_vs_cost_sensitive_training"][c] = test(
            aware, fold_cost(folds, "Cost-sensitive LightGBM", "default 0.5", r))
    return out


# ---------------------------------------------------------------- figures
def figure1(folds: pd.DataFrame, args, path: Path) -> None:
    """Fig. 1: (a) business cost of each decision strategy across cost ratios;
    (b) corrected 95% confidence intervals of the paired cost differences."""
    ratios = sorted(folds.ratio.unique())
    fig = plt.figure(figsize=(7.2, 3.5))
    a = fig.add_axes([0.07, 0.40, 0.26, 0.52])
    b = fig.add_axes([0.635, 0.12, 0.235, 0.80])

    # (a) mean cost per 1,000 firms; color, marker and dash all encode the strategy
    series = [("Cost-sensitive LightGBM", "default 0.5", "Cost-sensitive training", ORANGE, "D", "-"),
              ("LightGBM", "default 0.5", "Default 0.5 threshold", "#e87ba4", "s", (0, (3, 1.5))),
              ("SMOTE + LightGBM", "cost-aware", "SMOTE + cost-aware threshold", "#199e70", "^", "-"),
              ("LightGBM", "cost-aware", "Cost-aware threshold (L4)", BLUE, "o", "-")]
    for model, thr, label, color, marker, dash in series:
        cost = [fold_cost(folds, model, thr, r).mean() for r in ratios]
        a.plot(range(len(ratios)), cost, color=color, marker=marker, ms=5, lw=1.6, ls=dash,
               label=label, mec="white", mew=0.8, zorder=3)
    a.set_xticks(range(len(ratios)), [f"{r:g}:1" for r in ratios])
    a.set_xlim(-0.25, len(ratios) - 0.75)
    a.set_xlabel("Cost ratio c", fontsize=8)
    a.set_ylabel("Business cost per 1,000 firms", fontsize=8)
    a.set_title("(a) Cost by decision strategy", fontsize=8.5, loc="left", color=INK)
    a.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(-0.22, -0.2),
             handlelength=2.6, labelcolor=INK)

    # (b) mean paired difference A - B with its corrected 95% CI (negative = A cheaper)
    c_main, c_high, frac = args.cost_ratio, max(ratios), 1 / (args.folds - 1)
    aware = {r: fold_cost(folds, "LightGBM", "cost-aware", r) for r in ratios}
    default = {r: fold_cost(folds, "LightGBM", "default 0.5", r) for r in ratios}
    xbp_cost = fold_cost(folds, PROPOSED, "cost-aware", c_main)
    groups = [(f"XBP (10 ratios) − 65-ratio model, c = {c_main:g}",
               [(n, xbp_cost - fold_cost(folds, n, "cost-aware", c_main)) for n in
                ["Logistic regression", "Multilayer perceptron", "Random forest", "XGBoost", "LightGBM"]]),
              ("Decision strategies and SHAP consensus",
               [(f"Threshold − default, c = {c_main:g}", aware[c_main] - default[c_main]),
                (f"Threshold − default, c = {c_high:g}", aware[c_high] - default[c_high]),
                (f"Threshold − cost-sensitive, c = {c_high:g}",
                 aware[c_high] - fold_cost(folds, "Cost-sensitive LightGBM", "default 0.5", c_high)),
                (f"Threshold − SMOTE, c = {c_high:g}",
                 aware[c_high] - fold_cost(folds, "SMOTE + LightGBM", "cost-aware", c_high)),
                (f"Consensus − single SHAP, c = {c_high:g}",
                 fold_cost(folds, PROPOSED, "cost-aware", c_high)
                 - fold_cost(folds, f"Single-SHAP top-{args.k_main}", "cost-aware", c_high))])]
    y = 0
    for header, rows in groups:
        b.text(-0.03, y, header, transform=b.get_yaxis_transform(), ha="right", va="center",
               fontsize=7.5, color=MUTED, style="italic")
        y -= 1
        for label, diff in rows:
            m, lo, hi, _ = corrected_ttest(diff, frac)
            sig = hi < 0 or lo > 0
            color = BLUE if sig else MUTED
            b.plot([lo, hi], [y, y], color=color, lw=1.6, solid_capstyle="round", zorder=2)
            b.plot(m, y, "o", ms=5.5, color=color if sig else "white", mec=color, mew=1.4, zorder=3)
            b.text(-0.03, y, label, transform=b.get_yaxis_transform(), ha="right", va="center",
                   fontsize=7.5, color=INK)
            b.text(1.03, y, f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]".replace("-", "−"),
                   transform=b.get_yaxis_transform(), ha="left", va="center", fontsize=7, color=MUTED)
            y -= 1
        y -= 0.4
    b.axvline(0, color=INK, lw=0.8, zorder=1)
    b.set_yticks([]); b.set_ylim(y + 0.6, 0.6)
    b.set_xlabel("Cost difference per 1,000 firms (negative = first is cheaper)", fontsize=8)
    b.set_title("(b) Paired differences with 95% CI", fontsize=8.5, loc="left", color=INK, x=-0.95)
    b.spines["left"].set_visible(False)

    for ax in (a, b):
        ax.tick_params(labelsize=8, colors=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.grid(axis="y" if ax is a else "x", color=NEUTRAL, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def figure2(X: pd.DataFrame, y: np.ndarray, args, path: Path) -> pd.DataFrame:
    """Fig. 2: SHAP beeswarm of the top-k drivers of the final model fitted on all firms."""
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    pipe = make_pipeline(RedundancyFilter(), lightgbm(pos_weight, args.seed, args.jobs))
    full, _, inner_models = fit_with_threshold(pipe, X, y, [args.cost_ratio], inner)
    top = tuple(consensus_ranking([(full, X), *inner_models]).index[: args.k_main])
    model = make_pipeline(Select(top), lightgbm(pos_weight, args.seed, args.jobs)).fit(X, y)
    Z = X[list(top)]
    sv = shap.TreeExplainer(model[-1]).shap_values(Z)
    sv = sv[1] if isinstance(sv, list) else sv
    cmap = LinearSegmentedColormap.from_list("bo", [BLUE, NEUTRAL, ORANGE])
    plt.figure()
    shap.summary_plot(sv, Z, max_display=len(top), show=False, cmap=cmap, plot_size=(7.2, 4.4),
                      rng=np.random.default_rng(args.seed))   # seeds the beeswarm's dot jitter
    fig = plt.gcf()
    fig.axes[0].set_xlabel("SHAP value (impact on log-odds of failure)", fontsize=9)
    fig.axes[0].tick_params(labelsize=8.5)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    direction = [np.corrcoef(Z[c].rank(), sv[:, i])[0, 1] for i, c in enumerate(top)]
    return pd.DataFrame({"driver": top, "mean_abs_shap": np.abs(sv).mean(axis=0),
                         "higher_value_raises_risk": np.array(direction) > 0})


# ---------------------------------------------------------------- entry point
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=Path(__file__).with_name("data") / "bankruptcy.csv")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("results"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cost-ratio", type=float, default=10.0, help="cost(FN) / cost(FP)")
    ap.add_argument("--sensitivity", type=float, nargs="+", default=[5.0, 20.0])
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 20], help="driver counts")
    ap.add_argument("--k-main", type=int, default=10, help="driver count of the proposed XBP")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--quick", action="store_true", help="3x1 CV smoke test")
    args = ap.parse_args()
    if args.quick:
        args.folds, args.repeats = 3, 1
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    folds, stability, frequency = run(args)
    folds.to_csv(args.out / "fold_metrics.csv", index=False)
    t1 = table1(folds, args.cost_ratio, args.folds)
    t2 = table2(folds, args.k, args.k_main, stability, args.cost_ratio)
    t1.to_csv(args.out / "table1_benchmark.csv", index=False)
    t2.to_csv(args.out / "table2_ablation.csv", index=False)

    X, y = load_data(args.data)
    figure1(folds, args, args.out / "fig1_cost_effects")
    drivers = figure2(X, y, args, args.out / "fig2_shap_drivers")
    summary = {
        "firms": int(len(y)), "failures": int(y.sum()), "ratios_raw": int(X.shape[1]),
        "cv": f"{args.folds}x{args.repeats}", "cost_ratio": args.cost_ratio, "k_main": args.k_main,
        "driver_stability_jaccard": {k: round(v, 3) for k, v in stability.items()},
        "paired_tests": paired_tests(folds, args.k_main, args.folds),
        "specificity_range_table1": [round(float(v), 3) for v in folds[(folds.ratio == args.cost_ratio)
            & (folds.threshold == "cost-aware")].groupby("model").specificity.mean().loc[
            ["Logistic regression", "Multilayer perceptron", "Random forest", "XGBoost", "LightGBM",
             PROPOSED]].agg(["min", "max"])],
        "driver_selection_frequency": frequency.round(3).to_dict(),
        "final_drivers": drivers.round(4).to_dict(orient="records"),
        "runtime_min": round((time.time() - t0) / 60, 1),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\nTable 1\n", t1.to_string(index=False), "\n\nTable 2\n", t2.to_string(index=False))
    print(f"\nDone in {summary['runtime_min']} min -> {args.out}/")


if __name__ == "__main__":
    main()
